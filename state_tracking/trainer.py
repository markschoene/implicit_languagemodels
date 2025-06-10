import torch
from transformers import Trainer


class StateTrackingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # no change here
        self.compute_metrics = self._compute_metrics
        self.rel_diff = []
        self.steps = []

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Override to return (loss, predictions, labels) where
        predictions is already argmax(logits) on GPU.
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # forward
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            loss = outputs.loss if has_labels else None
            logits = outputs.logits

            # log implicit model metrics
            rel_diff = outputs.implicit_metrics.get("rel diff", None)
            if rel_diff is not None:
                self.rel_diff.append(rel_diff)
            steps = outputs.implicit_metrics.get("steps", None)
            if steps is not None:
                self.steps.append(steps)

        if prediction_loss_only:
            return loss, None, None

        # turn logits → token IDs on GPU, then move preds+labels to CPU
        preds = torch.argmax(logits, dim=-1).cpu()
        labels = inputs["labels"].cpu() if has_labels else None
        return loss, preds, labels

    def _compute_metrics(self, eval_pred):
        # eval_pred is now (predictions, labels) as NumPy arrays
        preds, labels = eval_pred
        correct = preds == labels

        # compute the mean accuracy of the final 10 tokens
        accuracy = correct[:, -10:].mean().item()

        # compute relative difference and steps
        if len(self.rel_diff) > 0:
            rel_diff = torch.stack(self.rel_diff).mean()
        else:
            rel_diff = None
        if len(self.steps) > 0:
            steps = torch.stack(self.steps).mean()
        else:
            steps = None

        # reset metrics for next evaluation
        self.rel_diff.clear()
        self.steps.clear()

        return {
            "next_token_accuracy": accuracy,
            "rel diff": rel_diff,
            "steps": steps,
        }
