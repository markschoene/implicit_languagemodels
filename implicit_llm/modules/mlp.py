# Copyright (c) 2024, Tri Dao, Albert Gu.
from enum import Enum
from functools import partial
from typing import Optional

from torch import Tensor, nn
from torch.nn import functional as F

from .dropout import VariationalDropout1d


class MLPType(Enum):
    """
    Enum class for the different types of MLPs that can be used in the model.
    """

    MLP = "MLP"
    GATED_MLP = "GATED_MLP"
    LLAMA_MLP = "LLAMA_MLP"


class MLP(nn.Module):
    """
    Standard transformer MLP.
    """

    def __init__(
        self,
        in_features: int,
        dropout=0.0,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation=F.gelu,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(4 * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

        self.drop1 = VariationalDropout1d(dropout=dropout)
        self.drop2 = VariationalDropout1d(dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = self.activation(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return y


class LlamaMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features=None,
        dropout=0.0,
        multiple_of: int = 1,  # Default to 1 if you don't need a specific alignment
        ffn_dim_multiplier: Optional[float] = None,
        bias: bool = False,  # Added bias parameter for flexibility
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        assert hidden_features is not None, "hidden_features must be specified for LlamaMLP"
        hidden_features = int(2 * hidden_features / 3)
        # Custom in_features factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_features = int(ffn_dim_multiplier * hidden_features)
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        # Using standard nn.Linear layers
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)  # fc2 naming is used for special initialization
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x):
        return self.fc2(F.silu(self.w1(x)) * self.w3(x))


class GatedMLP(nn.Module):
    """
    MLP with gated activation function.
    """

    def __init__(
        self,
        in_features,
        dropout=0.0,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

        self.drop1 = VariationalDropout1d(dropout=dropout)
        self.drop2 = VariationalDropout1d(dropout=dropout)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return y


def get_mlp_cls(mlp_type: str, **kwargs):
    """
    Get the MLP module based on the MLPType.
    """
    mlp_type = MLPType[mlp_type.upper()]
    if mlp_type == MLPType.MLP:
        return partial(MLP, **kwargs)
    elif mlp_type == MLPType.GATED_MLP:
        return partial(GatedMLP, **kwargs)
    elif mlp_type == MLPType.LLAMA_MLP:
        return partial(LlamaMLP, **kwargs)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")
