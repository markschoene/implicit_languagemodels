import json
import os
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn

from .modules.heads import PartialCrossEntropyHead


def gaussian_noise(
    tensor: torch.Tensor, snr_db: float, mode: str = "additive"
) -> torch.Tensor:
    """
    Apply Gaussian noise to a tensor (either additive or multiplicative) with a specified signal-to-noise ratio (SNR in dB).

    Args:
        tensor (torch.Tensor): The input tensor with shape (batch_size, hidden_size).
        snr_db (float): Signal-to-noise ratio in decibels (dB). Lower values correspond to higher noise.
        mode (str): The mode of noise ('additive' or 'multiplicative'). Default is 'additive'.

    Returns:
        torch.Tensor: The tensor with Gaussian noise applied.
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate the power of the signal (tensor)
    signal_power = tensor.pow(2)

    # Calculate the noise power based on the SNR (in linear scale)
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise with zero mean and the calculated standard deviation
    noise_std = torch.sqrt(noise_power)

    if mode == "additive":
        # Generate noise and add it to the tensor
        noise = torch.randn_like(tensor) * noise_std
        tensor.copy_(tensor + noise)
    elif mode == "multiplicative":
        # Generate multiplicative noise (1 + noise factor) and multiply with the tensor
        noise = torch.randn_like(tensor) * noise_std
        tensor.copy_(tensor * (1 + noise))
    else:
        raise ValueError("Mode must be either 'additive' or 'multiplicative'")


def apply_gaussian_noise(tree: Any, snr_db: float, mode: str = "additive") -> Any:
    """
    Recursively apply Gaussian noise to a tensor with a specified standard deviation.

    Args:
        tree (Any): The input data structure (tensor, dict, or list) to which Gaussian noise will be applied.
        snr_db (float): Signal-to-noise ratio in decibels (dB). Lower values correspond to higher noise.
        mode (str): The mode of noise ('additive' or 'multiplicative'). Default is 'additive'.

    Returns:
        torch.Tensor: The tensor with Gaussian noise applied.
    """
    if isinstance(tree, torch.Tensor):
        return gaussian_noise(tree, snr_db, mode)
    elif isinstance(tree, dict):
        return {k: gaussian_noise(v, snr_db, mode) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [gaussian_noise(item, snr_db, mode) for item in tree]
    else:
        raise TypeError("Unsupported type for Gaussian noise application.")


def load_checkpoint(pretrained_model_name_or_path: str) -> dict:
    """
    Load the checkpoint from the specified path. This function handles both sharded and non-sharded checkpoints.
    """
    # Determine whether we have a sharded checkpoint (using an index file)
    index_path = os.path.join(
        pretrained_model_name_or_path, "pytorch_model.bin.index.json"
    )
    if os.path.isfile(index_path):
        # Load the index file
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        state_dict = {}

        # Collect unique shard filenames from the index
        shard_files = set(weight_map.values())
        for shard_file in shard_files:
            shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Expected shard file {shard_path} not found.")
            # Load each shard and update our state_dict
            shard_state_dict = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_state_dict)
    else:
        # Otherwise, load a single checkpoint file.
        checkpoint_path = os.path.join(
            pretrained_model_name_or_path, "pytorch_model.bin"
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at {checkpoint_path}. Expected a file named pytorch_model.bin."
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return state_dict


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_iterations,
        lr_decay_iters=None,
        min_lr=None,
        decay_lr=True,
    ):
        self.warmup = warmup_steps
        self.max_num_iters = max_iterations
        self.decay_lr = decay_lr
        self.min_lr_coeff = 0.1  # per Chinchilla
        # use lr_decay_iters< max_iterations to start constant lr phase before max_iterations
        self.lr_decay_iters = (
            lr_decay_iters if lr_decay_iters is not None else max_iterations
        )
        self.min_lr = min_lr  # if None uses  min_lr_coeff of peak lr

        super().__init__(optimizer)

    def get_lr(self):

        def get_min_lr(base_lr):
            return (
                self.min_lr if self.min_lr is not None else base_lr * self.min_lr_coeff
            )

        if not self.decay_lr:
            return list(self.base_lrs)

        if self.last_epoch < self.warmup:
            # Linear warmup phase
            return [
                base_lr * self.last_epoch / self.warmup for base_lr in self.base_lrs
            ]
        elif self.last_epoch > self.lr_decay_iters:
            # Constant learning rate phase
            return [get_min_lr(base_lr) for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            decay_ratio = (self.last_epoch - self.warmup) / (
                self.lr_decay_iters - self.warmup
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1 + np.cos(np.pi * decay_ratio))
            return [
                (get_min_lr(base_lr)) + coeff * (base_lr - get_min_lr(base_lr))
                for base_lr in self.base_lrs
            ]


class LinearWarmupConstantDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_iterations, lr_decay_iters=None):
        self.warmup = warmup_steps
        self.max_num_iters = max_iterations
        self.cooldown_ratio = 0.2  # as per https://arxiv.org/pdf/2405.18392
        self.lr_decay_iters = (
            lr_decay_iters
            if lr_decay_iters is not None
            else int(max_iterations * (1 - self.cooldown_ratio))
        )
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            # Linear warmup phase
            return [
                base_lr * self.last_epoch / self.warmup for base_lr in self.base_lrs
            ]
        elif self.last_epoch <= self.lr_decay_iters:
            # Constant learning rate phase
            return list(self.base_lrs)
        else:
            # 1 - sqrt decay phase
            decay_ratio = (self.last_epoch - self.lr_decay_iters) / (
                self.max_num_iters - self.lr_decay_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 1 - np.sqrt(decay_ratio)
            return [base_lr * coeff for base_lr in self.base_lrs]


class Embedding(nn.Module):
    """Class that applies an extra projection to the input embeddings if the dimensions do not match"""

    def __init__(self, n_tokens, d_embed, d_model):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_tokens = n_tokens

        if d_model != d_embed:
            self.projection = nn.Linear(d_embed, d_model, bias=False)

        self.emb = nn.Embedding(n_tokens, d_embed)

    def forward(self, x):
        x = self.emb(x)
        if self.d_model != self.d_embed:
            x = self.projection(x)
        return x


class EmbeddingMixin:

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
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
            raise ValueError(
                "The model's head is not supported for resizing embeddings"
            )

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens, pad_to_multiple_of
        )
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
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:

        if pad_to_multiple_of is not None:
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = (
                (new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

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
        if (
            old_embeddings.padding_idx is not None
            and (new_num_tokens - 1) < old_embeddings.padding_idx
        ):
            old_embeddings.padding_idx = None

        return old_embeddings

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: Optional[int] = None,
        transposed: Optional[bool] = False,
    ) -> nn.Linear:

        if new_num_tokens is None:
            return old_lm_head

        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.size()
            if not transposed
            else old_lm_head.weight.t().size()
        )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        # Build new lm head
        new_lm_head_shape = (
            (old_lm_head_dim, new_num_tokens)
            if not transposed
            else (new_num_tokens, old_lm_head_dim)
        )
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
            new_lm_head,
            old_lm_head,
            num_tokens_to_copy,
            transposed,
            has_new_lm_head_bias,
        )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self,
        new_lm_head,
        old_lm_head,
        num_tokens_to_copy,
        transposed,
        has_new_lm_head_bias,
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
                :num_tokens_to_copy, :
            ]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[
                :, :num_tokens_to_copy
            ]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
                :num_tokens_to_copy
            ]
