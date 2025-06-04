import torch.nn as nn
import torchvision.models as models

def get_resnet_18(
        num_classes: int,
        img_size: int,
    ) -> nn.Module:
    model = models.resnet18(
        num_classes=num_classes,
    )
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
    )

    return model