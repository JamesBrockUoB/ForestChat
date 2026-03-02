from torchgeo.models import FCSiamDiff


class FCSiamDiff_Wrapper(FCSiamDiff):

    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights else None,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, imgA, imgB):
        import torch

        x = torch.stack([imgA, imgB], dim=1)
        return super().forward(x)
