import argparse

import torch
from transformers import AutoModel

import implicit_llm
from implicit_llm import sequential_forward

torch.manual_seed(0)


def simulataneous_forward(model, input_ids):
    """
    Perform a simultaneous forward pass through the model.
    """
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits


def test_evaluation(model, input_ids):
    # run model in sequential and simultaneous modes
    out_sim = simulataneous_forward(model, input_ids=input_ids)
    out_seq = sequential_forward(model, input_ids).logits

    # decode sequences
    dec_sim = out_sim.argmax(-1)
    dec_seq = out_seq.argmax(-1)
    print(f"Greedy decoded sequence (simultaneous):\n{dec_sim.detach().cpu().numpy()}")
    print(f"Greedy decoded sequence (sequential):\n{dec_seq.detach().cpu().numpy()}")


def prepare_model(model, device, precision):
    """
    Prepare the model for evaluation by moving it to the specified device and setting the precision.
    """
    model = model.to(device, precision)
    if hasattr(model, "backbone") and model.config.backbone_type == "implicit":
        model.backbone.pretrain = False
        model.backbone.tau = 0.8
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Path to model")

    args = parser.parse_args()
    batch_size = args.batch_size

    model = AutoModel.from_pretrained(args.model_name)

    device = "cuda"
    precision = torch.float32

    model = prepare_model(model, device, precision)

    input_ids = torch.randint(0, 240, (args.batch_size, 16)).to(device)

    print(40 * "=")
    print(f"Testing model {args.model_name}:")
    print(f"Model class: {model.__class__.__name__}")
    print(40 * "=")

    test_evaluation(model, input_ids)
