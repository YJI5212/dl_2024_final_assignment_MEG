import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm import create_model


class LightEnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(LightEnsembleModel, self).__init__()
        self.mobilenet = create_model("mobilenetv3_large_100", pretrained=True)
        self.resnet = create_model("resnet18", pretrained=True)

        # MobileNetV3とResNet18の出力特徴量のサイズを確認
        self.mobilenet.classifier = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.mobilenet_features = self._get_features(self.mobilenet)
        self.resnet_features = self._get_features(self.resnet)

        # 出力サイズに応じて調整
        self.fc = nn.Linear(self.mobilenet_features + self.resnet_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (b, 271, 281) -> (b, 1, 271, 281)
        x = x.repeat(1, 3, 1, 1)  # (b, 1, 271, 281) -> (b, 3, 271, 281)

        x1 = self.mobilenet(x)
        x2 = self.resnet(x)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

    def _get_features(self, model):
        dummy_input = torch.zeros(1, 3, 271, 281)
        features = model(dummy_input)
        return features.shape[1]


if __name__ == "__main__":
    from torchinfo import summary

    # モデルのインスタンスを作成
    num_classes = 1854
    seq_len = 281
    in_channels = 271
    model = LightEnsembleModel(num_classes=num_classes)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # モデル構造の表示
    summary(model, (1, in_channels, seq_len), device=device.type)
