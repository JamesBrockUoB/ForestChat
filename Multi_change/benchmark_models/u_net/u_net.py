import segmentation_models_pytorch as smp
import torch.nn as nn


class UNet(nn.Module):

    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights else None,
            in_channels=in_channels * 2,
            classes=classes,
        )

    def forward(self, imgA, imgB):
        import torch

        x = torch.cat([imgA, imgB], dim=1)
        return self.model(x)
