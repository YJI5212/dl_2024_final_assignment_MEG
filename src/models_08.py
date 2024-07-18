import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedMEG_RPSnet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, seq_len: int) -> None:
        super().__init__()

        self.block1 = ConvBlock(1, 64, kernel_size=(11, 11), stride=2, padding=5)
        self.block2 = ResidualBlock(64, 128, kernel_size=(9, 9), stride=2, padding=4)
        self.block3 = ResidualBlock(128, 256, kernel_size=(7, 7), stride=2, padding=3)
        self.block4 = ResidualBlock(256, 512, kernel_size=(7, 7), stride=2, padding=3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(
            1
        )  # Add a channel dimension: (batch_size, 1, in_channels, seq_len)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)  # Apply adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=2, padding=0, p_drop=0.5
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=2, padding=0, p_drop=0.5
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.conv_residual = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, padding=0
            )
        else:
            self.conv_residual = (
                nn.Identity()
            )  # Identity layer if in_channels == out_channels and stride == 1

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        residual = self.conv_residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # モデルのインスタンスを作成
    num_classes = 1854
    seq_len = 281
    in_channels = 271
    model = ImprovedMEG_RPSnet(
        num_classes=num_classes, in_channels=in_channels, seq_len=seq_len
    )

    # モデルのサマリを表示
    from torchsummary import summary

    summary(model, (in_channels, seq_len))

    # ダミーデータでのフォワードパス
    dummy_data = torch.randn(
        1, in_channels, seq_len
    )  # バッチサイズ1、in_channels、seq_len
    output = model(dummy_data)
    print(output)
