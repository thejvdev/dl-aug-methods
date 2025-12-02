import torch.nn as nn
from torchvision.models import resnet18
import kornia.augmentation as K


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.normalize = K.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, X):
        X = self.normalize(X)
        y_pred = self.model(X)
        return y_pred
