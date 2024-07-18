import torch
import torch.nn as nn
import torch.nn.functional as F


class MEG_RPSnet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, seq_len: int) -> None:
        super().__init__()

        self.block1 = ConvBlock(1, 64, kernel_size=(11, 11), stride=2, padding=5)
        self.block2 = ConvBlock(64, 128, kernel_size=(9, 9), stride=2, padding=4)
        self.block3 = ConvBlock(128, 256, kernel_size=(7, 7), stride=2, padding=3)
        self.block4 = ConvBlock(256, 512, kernel_size=(7, 7), stride=2, padding=3)

        # 畳み込み層の出力サイズを計算
        conv_output_size = self._get_conv_output_size(in_channels, seq_len)

        self.fc1 = nn.Linear(conv_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_output_size(self, in_channels, seq_len):
        # Calculate the output size of the final convolutional layer
        x = torch.rand(1, 1, in_channels, seq_len)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(
            1
        )  # Add a channel dimension: (batch_size, 1, in_channels, seq_len)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
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


if __name__ == "__main__":

    # モデルのインスタンスを作成
    num_classes = 1854
    seq_len = 281
    in_channels = 271
    model = MEG_RPSnet(
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
