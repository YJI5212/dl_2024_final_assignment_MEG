import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class BasicGRUClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128
    ):
        super(BasicGRUClassifier, self).__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hid_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hid_dim * 2, num_classes)  # bidirectionalなので2倍

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = X.permute(0, 2, 1)  # (b, c, t) -> (b, t, c)
        X, _ = self.gru(X)  # (b, t, 2*hid_dim)
        X = X[:, -1, :]  # 最後のタイムステップの出力を使用
        X = self.fc(X)  # (b, num_classes)

        return X


if __name__ == "__main__":
    from torchinfo import summary

    # モデルのインスタンスを作成
    num_classes = 1854
    seq_len = 281
    in_channels = 271
    model = BasicGRUClassifier(
        num_classes=num_classes, seq_len=seq_len, in_channels=in_channels
    )

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # モデル構造の表示
    summary(model, (1, in_channels, seq_len), device=device.type)
