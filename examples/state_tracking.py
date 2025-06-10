# coding: utf-8
import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments

import implicit_llm
from implicit_llm import sequential_forward
from state_tracking.dataset import HuggingFaceDataset
from state_tracking.trainer import StateTrackingTrainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to the evaluation dataset.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the testing dataset.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the model and results.")
    parser.add_argument("--eval", action="store_true", help="Don't train but only evaluate the model.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def load_dataset_auto(path, mode="train"):
    if os.path.exists(path):
        # Assume local file is a JSON or CSV, try to infer format
        if path.endswith(".bin"):
            dataset = HuggingFaceDataset(path)
            print(f"Loaded dataset from {path}")
            return dataset
        else:
            raise ValueError(f"Unsupported local file format: {path}")
    else:
        # Try to load from Hugging Face Hub
        return load_dataset(path)[mode]


def evaluate_sequentially(model, dataset, batch_size):
    """
    Evaluate the model in sequential evaluation mode on a state tracking dataset.
    Compare Figure 2B/C in the paper for an explanation of the sequential evaluation mode.
    """
    device = model.device
    model.eval()
    total_accuracy = 0.0
    total_loss = 0.0
    total_samples = 0
    rel_diffs = []
    steps = []

    is_implicit = model.config.backbone_type == "implicit"
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating model"):
            batch = dataset[i:i+batch_size]
            
            # Convert batch to tensors
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if is_implicit:
                outputs = sequential_forward(model, input_ids=input_ids)
            else:
                outputs = model(**batch, use_cache=False)
            
            logits = outputs.logits
            loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")

            # turn logits → token IDs on GPU, then move preds+labels to CPU
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).float()

            # compute the mean accuracy of the final 10 tokens
            total_accuracy += correct[:, -10:].mean().item() * len(input_ids)

            # Accumulate loss and sample count
            total_loss += loss.item() * len(input_ids)
            total_samples += len(input_ids)

            if is_implicit: # log implicit model metrics
                rel_diffs.append(outputs.implicit_metrics["rel diff"])
                steps.append(outputs.implicit_metrics["steps"])
    
    avg_accuracy = total_accuracy / total_samples
    avg_loss = total_loss / total_samples
    
    # print the metrics to terminal
    print(40 * "=")
    print(f"Average accuracy: {avg_accuracy:.4f}\nAverage loss: {avg_loss:.4f}")
    if is_implicit:
        rel_diff = torch.stack(rel_diffs).mean().item()
        steps_mean = torch.stack(steps).mean().item()
        print(f"Average relative difference: {rel_diff:.4f}\nAverage steps: {steps_mean:.4f}")
    print(40 * "=")


def main():
    """
    Main function that trains on a state tracking dataset.
    The models are read from the huggingface configs in `hf_configs`.
    """
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    train_dataset = args.train_dataset
    eval_dataset = args.eval_dataset
    test_dataset = args.test_dataset

    train_data = load_dataset_auto(train_dataset, mode="train")
    eval_data = load_dataset_auto(eval_dataset, mode="validation")
    test_data = load_dataset_auto(test_dataset, mode="test")

    if args.eval:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    else:
        cfg = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_config(cfg)

    # fixed unrolling for model warmup
    try:
        if model.config.backbone_type == "implicit":
            model.backbone.pretrain_steps = 4005  # 4k batches of warmup
            model.backbone.pretrain_iter = 4  # 4 self-iterations
            model.backbone.pretrain = True
    except AttributeError:
        pass

    training_args = TrainingArguments(
        run_name=args.model_name,
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=1,
        warmup_ratio=0.1,
        learning_rate=1e-3,
        adam_beta2=0.95,
        max_grad_norm=0.25,
        dataloader_num_workers=4,
        save_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        logging_dir=args.output_dir,
        logging_steps=100,
        report_to="tensorboard",
        save_safetensors=False,
        seed=args.seed,
    )

    trainer = StateTrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    if not args.eval:
        trainer.train()

    # Validation during training is conducted in simultaneous mode for optimal speed
    # For testing and demonstrating the duality between the simultaneous and sequential modes (see Figure 2B/C in the paper),
    # we evaluate the model in sequential mode on the test set.
    evaluate_sequentially(model, test_data, args.batch_size)


if __name__ == "__main__":
    main()
