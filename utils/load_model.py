import torch
from .model import ResNet18


def load_model(checkpoint_path, device="cpu"):
    model = ResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer
