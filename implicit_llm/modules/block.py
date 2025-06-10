# Copyright (c) 2024, Tri Dao, Albert Gu.
import copy
from enum import Enum
from functools import partial
from typing import Any

import torch
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor, nn

from ..modules.attention import CausalSelfAttention
from ..modules.mamba2 import Mamba2
from ..modules.mlp import get_mlp_cls


class BlockType(Enum):
    """
    Enum class for the different types of blocks that can be used in the model.
    """

    MAMBA2 = "mamba2"
    TRANSFORMER = "transformer"


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        mixer_cls,
        mlp_cls,
        norm_cls=nn.LayerNorm,
        pre_norm: bool = False,
        residual_in_fp32: bool = True,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.residual_in_fp32 = residual_in_fp32
        self.norm = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not None:
            self.norm2 = norm_cls(d_model)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

    def forward(
        self,
        zs: Tensor,
        injected_inputs: Tensor | None = None,
        **mixer_kwargs,
    ) -> Tensor:
        # split off residual
        residual = zs

        # pre-norm formulation
        if self.pre_norm:
            zs = self.norm(zs.to(dtype=self.norm.weight.dtype))

        # apply the time mixer (SSM / Transformer) and skip connection
        zs = self.mixer(zs, injected_inputs, **mixer_kwargs)
        zs = residual + zs
        if self.residual_in_fp32:
            zs = zs.to(torch.float32)

        # post-norm
        if not self.pre_norm:
            zs = self.norm(zs.to(dtype=self.norm.weight.dtype))

        # apply the MLP if an MLP class is given
        if self.mlp is not None:
            residual = zs

            # pre-norm formulation
            if self.pre_norm:
                zs = self.norm2(zs.to(dtype=self.norm2.weight.dtype))

            # apply channel mixer (Gated MLP) and skip connection
            zs = self.mlp(zs)
            zs = residual + zs
            if self.residual_in_fp32:
                zs = zs.to(torch.float32)

            # post-norm formulation
            if not self.pre_norm:
                zs = self.norm2(zs.to(dtype=self.norm2.weight.dtype))

        return zs


def create_block(
    block_type: str,
    d_model: int,
    d_intermediate: int,
    block_cfg: dict[str, Any] | None = None,
    dropout: float = 0.0,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    pre_norm: bool = False,
    residual_in_fp32: bool = True,
    mlp_type: str = "gated",
    layer_idx=None,
    device=None,
    dtype=None,
) -> Block:
    """
    Create a block with the given configuration.

    Arguments
    ---------
        block_type can be one of "mamba2" or "transformer".
    """
    if block_cfg is None:
        block_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # create SSM module
    # Create a copy of the config to modify
    block_cfg = copy.deepcopy(block_cfg) if block_cfg is not None else {}
    mixer_init = CausalSelfAttention if BlockType[block_type.upper()] == BlockType.TRANSFORMER else Mamba2
    mixer_cls = partial(mixer_init, layer_idx=layer_idx, dropout=dropout, **block_cfg, **factory_kwargs)

    # Normalization layers
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)

    # Feed forward network
    if d_intermediate == 0:
        mlp_cls = None
    else:
        mlp_cls = get_mlp_cls(
            mlp_type,
            hidden_features=d_intermediate,
            out_features=d_model,
            dropout=dropout,
            **factory_kwargs,
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        pre_norm=pre_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block
