"""
Embedding-related modules.
"""

import torch
from torch import nn

from .heads import PartialCrossEntropyHead


class EmbeddingMixin:
    """
    Enable embedding resizing and weight tying for models that use embeddings.
    """

    def resize_token_embeddings(
        self, new_num_tokens: int | None = None, pad_to_multiple_of: int | None = None
    ) -> nn.Embedding:
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def get_input_embeddings(self):
        return self.word_emb.emb

    def set_input_embeddings(self, new_embeddings):
        self.word_emb.emb = new_embeddings

    def get_output_embeddings(self) -> nn.Module:
        if isinstance(self.criterion, PartialCrossEntropyHead):
            return self.criterion.decoder
        else:
            return None

    def set_output_embeddings(self, new_embeddings):
        if isinstance(self.criterion, PartialCrossEntropyHead):
            self.criterion.decoder = new_embeddings
        else:
            raise ValueError("The model's head is not supported for resizing embeddings")

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.tie_embeddings:
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)

            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> nn.Embedding:
        if pad_to_multiple_of is not None:
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings

        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        old_embeddings.weight.data = new_embeddings.weight.data
        old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
        if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
            old_embeddings.padding_idx = None

        return old_embeddings

    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: int | None = None, transposed: bool | None = False
    ) -> nn.Linear:
        if new_num_tokens is None:
            return old_lm_head

        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        self._copy_lm_head_original_to_resized(
            new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
        )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]


class Embedding(nn.Module):
    """Class that applies an extra projection to the input embeddings if the dimensions do not match"""

    def __init__(self, vocab_size: int, d_embed: int, d_model: int):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.vocab_size = vocab_size

        if d_model != d_embed:
            self.projection = nn.Linear(d_embed, d_model, bias=False)

        self.emb = nn.Embedding(vocab_size, d_embed)

    def forward(self, x):
        x = self.emb(x)
        if self.d_model != self.d_embed:
            x = self.projection(x)
        return x
