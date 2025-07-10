import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import implicit_llm
from implicit_llm import sequential_forward
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# run
# python noise.py --model_name hf_models/mamba2-130m-implicit --batches 100 --batch_size 128 --noise_mode additive --state_noise_db 20.0 --latent_noise_db 20.0 --ema_alpha 0.05


def load_and_tokenize_dataset(tokenizer, dataset, max_length):

    tokenized_data = dataset.map(
        lambda e: tokenizer(
            e["text"], truncation=True, padding="max_length", max_length=max_length
        ),
        batched=True,
    )
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def compute_length(batch):
        # For each sample, compute the length of the tokenized 'input_ids'
        return {"length": [(ids != 0).sum() for ids in batch["input_ids"]]}

    # Apply the function in batch mode to speed up processing
    tokenized_data_with_length = tokenized_data.map(compute_length, batched=True)

    return tokenized_data_with_length


def dataset_statistics(dataset):
    lengths = np.zeros(len(dataset))
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        lengths[i] = sample["length"]
    print("Average length per sample:", np.mean(lengths))
    print("Max length per sample:", np.max(lengths))
    print("Median length per sample:", np.median(lengths))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(lengths, bins=np.linspace(0, lengths.max(), 20), alpha=0.7, color="blue")
    ax.set_title("Length Distribution of Samples")
    ax.set_xlabel("Length")
    ax.set_ylabel("Frequency")
    fig.savefig("noise_results/img/data_length_distribution.png", dpi=300)


def apply_noise_params(model, noise_mode, state_noise_db, latent_noise_db, ema_alpha):
    """
    Apply noise parameters to the model's layers.
    """
    if model.config.backbone_type == "implicit":
        model.backbone.ema_alpha = ema_alpha

    for layer in model.backbone.layers:
        layer.mixer.noise_mode = noise_mode
        layer.mixer.state_noise_db = state_noise_db
        layer.mixer.latent_noise_db = latent_noise_db


def evaluation_step(model, batch):
    input_ids = batch["input_ids"].to("cuda")  # Move to GPU
    attention_mask = batch["attention_mask"].to("cuda")  # Move to GPU

    # Shift labels to the left by 1 for causal language modeling, pad last token with -100 to ignore in loss
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100

    # Forward pass
    outputs = sequential_forward(model, input_ids=input_ids)

    logits = outputs.logits  # (batch, seq_len, vocab)
    # Compute log-probs
    log_probs = torch.log_softmax(logits, dim=-1)
    # Only gather log-probs at valid label positions (labels != -100)
    valid_mask = labels != -100
    # For gather, set invalid label positions to any valid index (e.g., 0), will be masked out later
    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0
    nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(
        -1
    )  # (batch, seq_len)
    # Mask out ignored tokens
    nll = nll * valid_mask * attention_mask

    # Compute accuracy per token
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) * attention_mask.bool()
    accuracy = correct.float()

    # empty memory
    del outputs, logits, log_probs
    torch.cuda.empty_cache()

    return nll.cpu(), accuracy.cpu(), (attention_mask * valid_mask).cpu()


def compute_perplexity_and_accuracy(model, dataloader, terminate):
    all_perplexities = []
    all_accuracies = []
    all_mask = []

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if i >= terminate:
                break

            nll, accuracy, mask = evaluation_step(model, batch)
            all_perplexities.append(nll.cpu())
            all_mask.append(mask.cpu())
            all_accuracies.append(accuracy.cpu())

    # Stack to get (N, L)
    perplexity_tensor = torch.cat(all_perplexities, dim=0)
    accuracy_tensor = torch.cat(all_accuracies, dim=0)
    mask_tensor = torch.cat(all_mask, dim=0)

    return perplexity_tensor, accuracy_tensor, mask_tensor


def visualize_results(
    title, perplexity_tensor, accuracy_tensor, mask_tensor, avg_bins=8
):
    valid_mask = mask_tensor.sum(dim=0)
    # Compute average perplexity and accuracy per token position, ignoring masked positions
    avg_perplexity_per_pos = (perplexity_tensor * mask_tensor).sum(dim=0) / valid_mask
    avg_accuracy_per_pos = (accuracy_tensor * mask_tensor).sum(dim=0) / valid_mask

    avg_perplexity_per_pos = avg_perplexity_per_pos[valid_mask > 0]
    avg_accuracy_per_pos = avg_accuracy_per_pos[valid_mask > 0]

    print("Average perplexity per token position:", avg_perplexity_per_pos)
    print("Average accuracy per token position:", avg_accuracy_per_pos)

    print(
        "Number of nan perplexities:", torch.isnan(avg_perplexity_per_pos).sum().item()
    )

    # ensure that avg_perplexity_per_pos and avg_accuracy_per_pos have length divisible by avg_bins
    if len(avg_perplexity_per_pos) % avg_bins != 0:
        avg_perplexity_per_pos = avg_perplexity_per_pos[
            : len(avg_perplexity_per_pos) // avg_bins * avg_bins
        ]
    if len(avg_accuracy_per_pos) % avg_bins != 0:
        avg_accuracy_per_pos = avg_accuracy_per_pos[
            : len(avg_accuracy_per_pos) // avg_bins * avg_bins
        ]
    avg_perplexity_per_pos = (
        avg_perplexity_per_pos.numpy().reshape(avg_bins, -1).mean(axis=0)
    )
    avg_accuracy_per_pos = (
        avg_accuracy_per_pos.numpy().reshape(avg_bins, -1).mean(axis=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(
        np.arange(len(avg_perplexity_per_pos)) * avg_bins,
        np.exp(avg_perplexity_per_pos),
        label="Perplexity",
    )
    axes[0].set_title("Average Perplexity per Token Position")
    axes[0].set_xlabel("Token Position")
    axes[0].set_ylabel("Perplexity")
    axes[0].set_ylim(0, 100)  # Set y-axis limit for better visibility
    axes[0].legend()
    axes[1].plot(
        np.arange(len(avg_accuracy_per_pos)) * avg_bins,
        avg_accuracy_per_pos,
        label="Accuracy",
        color="orange",
    )
    axes[1].set_title("Average Accuracy per Token Position")
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)  # Set y-axis limit for better visibility
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"noise_results/img/{title.replace(' - ', '-')}.png", dpi=300)

    # calculate average perplexity and accuracy without NaN values
    avg_perplexity_per_pos = torch.tensor(avg_perplexity_per_pos, dtype=torch.float32)
    avg_accuracy_per_pos = torch.tensor(avg_accuracy_per_pos, dtype=torch.float32)
    avg_perplexity_per_pos[torch.isnan(avg_perplexity_per_pos)] = 999999
    avg_accuracy_per_pos[torch.isnan(avg_accuracy_per_pos)] = 0.0
    return avg_accuracy_per_pos.mean().item(), avg_perplexity_per_pos.mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Compute perplexity and accuracy for a model."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Huggingface model name or path"
    )
    parser.add_argument(
        "--batches", type=int, default=1, help="Number of batches to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Mode of noise to apply (additive or multiplicative)",
    )
    parser.add_argument(
        "--state_noise_db", type=float, default=20.0, help="State noise in dB"
    )
    parser.add_argument(
        "--latent_noise_db", type=float, default=20.0, help="Latent noise in dB"
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.05,
        help="Exponentially moving average for latent iteration",
    )
    args = parser.parse_args()

    # make directories noise_results/img and noise_results/data if they do not exist
    os.makedirs("noise_results/img", exist_ok=True)
    os.makedirs("noise_results/data", exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    apply_noise_params(
        model,
        noise_mode=args.noise_mode,
        state_noise_db=args.state_noise_db,
        latent_noise_db=args.latent_noise_db,
        ema_alpha=args.ema_alpha,
    )
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("allenai/C4", "realnewslike", split="validation")

    tokenized_data = load_and_tokenize_dataset(tokenizer, dataset, max_length=1024)
    dataloader = DataLoader(tokenized_data, batch_size=args.batch_size)

    # collect statistics about the dataset
    dataset_statistics(tokenized_data)

    # compute perplexity and accuracy
    perplexity_tensor, accuracy_tensor, mask_tensor = compute_perplexity_and_accuracy(
        model, dataloader, terminate=args.batches
    )
    title = f"{os.path.basename(args.model_name)} - {args.noise_mode} - {args.state_noise_db}dB - {args.latent_noise_db}dB - {args.ema_alpha}"
    avg_acc, avg_ppl = visualize_results(
        title, perplexity_tensor, accuracy_tensor, mask_tensor
    )

    # Save results to a JSON file
    print(f"Average Accuracy: {avg_acc:.4f}, Average Perplexity: {avg_ppl:.4f}")
    results = {
        "model_name": args.model_name,
        "noise_mode": args.noise_mode,
        "state_noise_db": args.state_noise_db,
        "latent_noise_db": args.latent_noise_db,
        "ema_alpha": args.ema_alpha,
        "average_accuracy": avg_acc,
        "average_perplexity": avg_ppl,
    }
    results_file = f"noise_results/data/{title.replace(' - ', '-')}.json"
    json.dump(results, open(results_file, "w"), indent=4)


if __name__ == "__main__":
    main()
