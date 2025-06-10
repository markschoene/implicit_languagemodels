from torchdeq.dropout import _VariationalDropoutNd
import torch


# need this class as the torchdeq implementation places the dropout mask on the CPU
# this leads to problems when training on GPU
class VariationalDropout1d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zero out the entire channel/feature dimension of the input 1d tensor
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The channel/feature dimension of 1d tensor is the :math:`*` slice of :math:`(B, L, *)`
    for token_first=True, or :math:`(B, *, L)` for token_first=False.

    The same mask is used for each input in a training iteration. (for fixed point convergence)
    This random mask is reset at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expects input tensor in shape :math:`(B, L, D)`,
                                       otherwise expects :math:`(B, D, L)`. Here, `B` is batch size,
                                       `L` is sequence length, and `D` is feature dimension.
                                       Default: False.

    Shape:
        - Input: :math:`(B, L, D)` or :math:`(B, D, L)`.
        - Output: :math:`(B, L, D)` or :math:`(B, D, L)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, L, D)
            B, _, D = x.shape
            m = torch.zeros(B, 1, D).bernoulli_(1 - self.dropout)
        else:
            # Dimension (B, D, L)
            B, D, _ = x.shape
            m = torch.zeros(B, D, 1).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask.to(x)
