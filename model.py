import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class OrientationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features,
            1,
        )

    def forward(self, x):
        return self.model(x).squeeze(1)