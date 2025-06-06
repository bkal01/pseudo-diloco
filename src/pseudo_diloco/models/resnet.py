import torch
from torchvision.models import resnet18

def get_resnet_18(
        num_classes: int,
) -> torch.nn.Module:
    model = resnet18(
        num_classes=num_classes,
    )
    # Replace normal ResNet downsampling with stride 1 convolution.
    model.conv1 = torch.nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = torch.nn.Identity()
    return model