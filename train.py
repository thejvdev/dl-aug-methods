import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import torchvision.transforms as T
from torchvision.transforms.v2 import MixUp
from torchvision.datasets import CIFAR10

import kornia.augmentation as K
from kornia.geometry import resize

from utils.model import ResNet18
from utils.early_stopping import EarlyStopping

from augmentation.fmix import FMix
from augmentation.agmix import AGMix


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Device:", device)

    train_ds = CIFAR10("data", train=True, transform=T.ToTensor(), download=True)

    max_lr = 1.5e-3
    epochs = 20

    method = "mixup"

    k = 5
    skf = StratifiedKFold(k, shuffle=True, random_state=42)
    targets = np.array(train_ds.targets)

    scaler = torch.amp.GradScaler("mps")

    print(f"Training...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_ds.data, targets), start=1
    ):
        # Data
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(train_ds, val_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

        # Train setup
        model = ResNet18()
        model.to(device)

        optimizer = AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # 1cycle policy
        steps = len(train_loader) * epochs

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=steps,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

        # Early Stopping
        early_stopping = EarlyStopping(
            patience=8,
            path=f"models/{method}/best_model_fold{fold}.pt",
            save_optimizer=True,
        )

        # Augmentations
        simple_aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=10, p=0.3),
            K.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.8),
            K.RandomGaussianNoise(mean=0.0, std=0.005, p=0.2),
            same_on_batch=False,
        )
        simple_aug = simple_aug.to(device)

        # hard_aug = FMix(num_classes=10)
        # hard_aug = AGMix(num_classes=10)
        hard_aug = MixUp(alpha=0.2, num_classes=10)

        writer = SummaryWriter(log_dir=f"runs/exp{fold}")

        print(f"Fold [{fold}/{k}]")

        # Train loop
        for epoch in range(epochs):
            model.train()

            train_loss = torch.tensor(0.0, device=device)
            train_correct = 0
            train_total = 0

            for X, y in train_loader:
                optimizer.zero_grad(set_to_none=True)

                X, y = X.to(device), y.to(device)

                with torch.autocast(device_type="mps", dtype=torch.float16):
                    X = resize(X, (224, 224))
                    X = simple_aug(X)
                    X, y = hard_aug(X, y)
                    y_pred = model(X)
                    loss = criterion(y_pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                # Metrics
                batch_size = X.size(0)
                train_loss += loss.detach() * batch_size
                # train_correct += (y_pred.argmax(1) == y).sum()
                train_correct += (y_pred.argmax(1) == y.argmax(1)).sum()
                train_total += batch_size

            # Validation
            model.eval()

            val_loss = torch.tensor(0.0, device=device)
            val_correct = 0
            val_total = 0

            with torch.inference_mode(), torch.autocast(
                device_type="mps", dtype=torch.float16
            ):
                for X, y in val_loader:
                    X = resize(X.to(device), (224, 224))
                    y = y.to(device)

                    y_pred = model(X)
                    loss = criterion(y_pred, y)

                    # Metrics
                    batch_size = X.size(0)
                    val_loss += loss * batch_size
                    val_correct += (y_pred.argmax(1) == y).sum()
                    val_total += batch_size

            train_loss = (train_loss / train_total).item()
            train_acc = (train_correct.float() / train_total).item()
            val_loss = (val_loss / val_total).item()
            val_acc = (val_correct.float() / val_total).item()

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)

            early_stopping(val_loss, model, optimizer)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break


if __name__ == "__main__":
    main()
