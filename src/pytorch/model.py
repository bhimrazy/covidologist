import torch
import torchvision
import torch.nn as nn
from src.pytorch.config import DEVICE


class DenseNet121(nn.Module):
    """DenseNet121 pretrained model definition."""

    def __init__(self, out_size: int):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(
            weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features*2),
            nn.Dropout(0.5),
            nn.Linear(in_features*2, in_features),
            nn.Dropout(0.30),
            nn.Linear(in_features, in_features//2),
            nn.Dropout(0.25),
            nn.Linear(in_features//2, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


model = DenseNet121(out_size=2).to(DEVICE)

# model configurations

criterion = torch.nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)
