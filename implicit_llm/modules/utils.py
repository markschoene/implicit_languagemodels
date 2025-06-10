from typing import Optional

import torch
from torch import Tensor


def compute_sequence_loss(
    evaluate_tokens: Tensor,
    target: Tensor,
    loss_fn: torch.nn.Module,
    vocab_size: int,
    mask: Tensor | None = None,
) -> Tensor:
    if mask is not None:
        if mask.sum().item() == 0:
            loss = torch.tensor(0.0).to(mask.device)
        else:
            masked_predict = evaluate_tokens[mask]
            masked_target = target[mask].view(-1)
            # if all target is ignore-index, prevent division by zero and NaN loss
            if len(masked_target) == 0 or (masked_target == loss_fn.ignore_index).all():
                loss = torch.tensor(0.0).to(mask.device)
            else:
                loss = loss_fn(masked_predict, masked_target)
    else:
        # explicitly make float due to edge case: if target is ALL masked output is long zero
        loss = loss_fn(evaluate_tokens.reshape(-1, vocab_size), target.flatten())
    return loss


def get_score(eval_task, n_gram, x_decode, pred_decode):
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    x_out = x_decode.split(".")[0] + "."

    if eval_task == "prefix_ngram":
        index = find(x_out, "|")[-1]
    elif eval_task in ["suffix_ngram", "copy", "duplicate_ngram"]:
        index = x_out.index("|")

    if eval_task == "suffix_ngram":
        gt = x_out[index + 1 + n_gram :][:-1]
        start_idx = index + n_gram
    else:
        gt = x_out[index + 1 :][:-1]
        start_idx = index

    end_idx = start_idx + len(gt)
    # gt is based on x rather than target so shift is not needed
    pred_model = pred_decode[start_idx:end_idx]

    str_acc = int(gt == pred_model)
    char_acc = sum(map(str.__eq__, gt, pred_model)) / max(len(gt), len(pred_model))

    return str_acc, char_acc


def get_mask_accuracy(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    mask = mask.view(-1)

    masked_pred = pred[mask.bool()]
    masked_target = target[mask.bool()]

    correct_predictions = (masked_pred == masked_target).float()
    accuracy = correct_predictions.mean()

    return accuracy


def get_string_accuracy(output_ids: Tensor, targets: Tensor, mask: Optional[Tensor] = None) -> float:
    """
    Full-string accuracy averaged over the batch.
    """
    bsz_size = output_ids.size(0)
    all_correct = 0.0
    for b in range(bsz_size):
        if mask is not None:
            output_ids_masked = output_ids[b, mask[b].bool()]
            targets_masked = targets[b, mask[b].bool()]
            all_correct += float((output_ids_masked == targets_masked).all())
        else:
            all_correct += float((output_ids[b] == targets[b]).all())
    return all_correct / bsz_size


def get_pred_metrics(logits: Tensor, target: Tensor, mask: Tensor) -> tuple[float, float]:
    """
    Computes accuracy and string accuracy for the given logits and target.
    Args:
        logits: The predicted logits from the model.
        target: The ground truth target tensor.
        mask (optional): The mask tensor to filter out certain tokens.

    Returns:
        - acc: The accuracy of the predictions.
        - string_accuracy: The string accuracy of the predictions.
    """
    pred = logits.argmax(dim=-1)
    if mask is not None:
        acc = get_mask_accuracy(pred, target, mask)
        string_accuracy = get_string_accuracy(pred, target, mask)
    else:
        # Average over all tokens
        acc = (pred == target).float().mean()
        string_accuracy = get_string_accuracy(pred, target)

    return acc, string_accuracy
