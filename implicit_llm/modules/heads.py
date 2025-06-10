"""
This module contains Loss functions. Note, that PartialCrossEntropyHead
actually contains the final projection layer (decoder) to project the model
down to the vocabulary size. This is was due to us initially mirroring the transfomer xl repo.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..modules.utils import compute_sequence_loss, get_pred_metrics


class PartialCrossEntropyHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_model: int,
        tokenizer=None,
        head_bias: bool = False,
        ignore_index: int = -100,
    ):
        super(PartialCrossEntropyHead, self).__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

        # this attribute is called by the lightning module to distinguish the computation of accuracies
        self.accuracy = True

        if d_model != d_embed:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_embed, bias=False),
                nn.Linear(d_embed, vocab_size, bias=True if head_bias else False),
            )
        else:
            self.decoder = nn.Linear(d_embed, vocab_size, bias=True if head_bias else False)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self, logits: Tensor, target: Tensor | None = None, mask: Tensor | None = None, just_decode: bool = False
    ) -> tuple[Tensor, float, float, Tensor]:

        if target is None and just_decode:
            evaluate_tokens: Tensor = self.decoder(logits)
            return None, None, None, evaluate_tokens

        if target is None:
            target = torch.zeros_like(logits)[..., 0].to(logits.device).long()
        bs, sample_length, dims = logits.size()
        bs, target_length = target.size()

        # evaluate only the last target_length tokens
        assert sample_length >= target_length, "Sample length must be greater than target length"
        evaluate_logits = logits[:, -target_length:, :]

        evaluate_tokens: Tensor = self.decoder(evaluate_logits)

        # compute the accuracy
        token_accuracy, string_accuracy = get_pred_metrics(
            evaluate_tokens,
            target,
            mask,
        )

        # compute the cross entropy loss
        loss = compute_sequence_loss(evaluate_tokens, target, self.loss_fn, self.vocab_size, mask)

        return loss, token_accuracy, string_accuracy, evaluate_tokens

    def tie_weights(self, embeddings: nn.Embedding):
        """
        Tie the weights of the final projection layer to the input embeddings.
        """
        num_embeddings, embedding_dim = embeddings.emb.weight.shape

        if self.d_model != self.d_embed:
            # the decoder is a nn.Sequential, so we need to set the weight of the first layer
            self.decoder[0].weight.data = embeddings.projection.weight.t()
            self.decoder[1] = nn.Linear(embedding_dim, num_embeddings, bias=self.decoder.bias is not None)
            self.decoder[1].weight = embeddings.emb.weight
        else:
            self.decoder = nn.Linear(embedding_dim, num_embeddings, bias=self.decoder.bias is not None)
            self.decoder.weight = embeddings.emb.weight
        self.vocab_size = num_embeddings
