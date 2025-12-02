import os
import torch
from typing import Optional


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        path: str = "checkpoint.pt",
        save_optimizer: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.path = path
        self.save_optimizer = save_optimizer

        self.counter = 0
        self.early_stop = False
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def __call__(self, value, model, optimizer: Optional[torch.optim.Optimizer] = None):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        else:
            value = float(value)

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self._save(model, optimizer)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model, optimizer):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        checkpoint = {"model_state_dict": model.state_dict()}
        if self.save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, self.path)
