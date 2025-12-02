import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from kornia.geometry import resize
from utils.model import ResNet18

import matplotlib.pyplot as plt


def find_lr(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
    smooth_f: float = 0.0,
    diverge_factor: float = 5.0,
):
    if num_iter < 2:
        raise ValueError("num_iter must be >= 2")

    was_training = model.training

    # Save states
    model_state = copy.deepcopy(model.state_dict())
    optim_state = copy.deepcopy(optimizer.state_dict())

    # Set initial LR
    lr = start_lr
    for g in optimizer.param_groups:
        g["lr"] = lr

    lr_mult = (end_lr / start_lr) ** (1.0 / (num_iter - 1))

    model.train()
    lrs, losses = [], []
    best_loss = None

    data_iter = iter(train_loader)

    for _ in range(num_iter):
        try:
            X, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            X, y = next(data_iter)

        X = resize(X.to(device), (224, 224))
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss_val = loss.item()

        # Optional smoothing
        if smooth_f > 0 and len(losses) > 0:
            loss_val = smooth_f * loss_val + (1 - smooth_f) * losses[-1]

        loss.backward()
        optimizer.step()

        # Log real LR
        lrs.append(lr)
        losses.append(loss_val)

        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val

        # Early stop
        if loss_val > best_loss * diverge_factor:
            break

        # Increase LR
        lr *= lr_mult
        for g in optimizer.param_groups:
            g["lr"] = lr

    # Restore original state
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    model.train() if was_training else model.eval()

    # Convert to tensors
    lrs_t = torch.tensor(lrs)
    losses_t = torch.tensor(losses)

    # Find candidate lr_min / lr_max (same logic as before)
    if len(losses_t) < 5:
        idx_min = torch.argmin(losses_t).item()
        return lrs_t[max(0, idx_min // 10)].item(), lrs_t[idx_min].item()

    log_lrs = torch.log10(lrs_t)
    n = len(losses_t)
    start = max(1, int(0.05 * n))
    end = max(start + 2, int(0.9 * n))

    sub_losses = losses_t[start:end]
    sub_log_lrs = log_lrs[start:end]

    d_loss = sub_losses[1:] - sub_losses[:-1]
    d_loglr = sub_log_lrs[1:] - sub_log_lrs[:-1]

    grads = d_loss / d_loglr

    idx_min_rel = torch.argmin(grads).item() + 1
    idx_min = start + idx_min_rel
    lr_min = lrs_t[idx_min].item()

    post_losses = losses_t[idx_min:end]
    post_lrs = lrs_t[idx_min:end]
    idx_max_rel = torch.argmin(post_losses).item()
    lr_max = post_lrs[idx_max_rel].item()

    plt.figure(figsize=(8, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss (log scale)")
    plt.title("LR Finder Curve")
    plt.grid(True, alpha=0.3)

    # Show markers for lr_min and lr_max
    plt.axvline(lr_min, color="green", linestyle="--", label=f"lr_min={lr_min:.2e}")
    plt.axvline(lr_max, color="red", linestyle="--", label=f"lr_max={lr_max:.2e}")
    plt.legend()
    plt.savefig("plots/lr_range.png")

    return lr_min, lr_max


def main():
    os.makedirs(f"plots", exist_ok=True)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Device:", device)

    train_ds = CIFAR10("data", train=True, transform=T.ToTensor(), download=True)

    n = min(len(train_ds), 2048)
    idxs = torch.randperm(len(train_ds))[:n]
    subset = Subset(train_ds, idxs)
    loader = DataLoader(subset, batch_size=128, shuffle=True)

    model = ResNet18()
    model.to(device)

    optimizer = AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print(f"Finding LR range...")

    min_lr, max_lr = find_lr(
        model, optimizer, criterion, train_loader=loader, device=device, smooth_f=0.1
    )
    print(f"LR Range: [{min_lr}, {max_lr}]")


if __name__ == "__main__":
    main()
