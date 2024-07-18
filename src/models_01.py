import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm import create_model


class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        """ self.efficientnet = create_model("efficientnet_b0", pretrained=True)
        self.regnety = create_model("regnety_008", pretrained=True) """
        self.efficientnet = create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )
        self.regnety = create_model(
            "regnety_008", pretrained=True, num_classes=num_classes
        )

        self.efficientnet.classifier = nn.Identity()  # 最終層を無効化
        self.regnety.head.fc = nn.Identity()  # 最終層を無効化

        # 出力サイズに応じて調整
        self.fc = nn.Linear(
            self.efficientnet.num_features + self.regnety.num_features, num_classes
        )

    def forward(self, x):
        # 1次元データを2次元に変換
        x = x.unsqueeze(1)  # (b, 128, 281) -> (b, 1, 128, 281)
        x = x.repeat(1, 3, 1, 1)  # (b, 1, 128, 281) -> (b, 3, 128, 281)

        x1 = self.efficientnet(x)
        x2 = self.regnety(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
