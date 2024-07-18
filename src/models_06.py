import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        # super(WaveNetBlock, self).__init__()
        super().__init__()
        # self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv_filter = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.conv_gate = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.residual = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, X):
        filter_out = self.conv_filter(X)[:, :, : X.size(2)]
        gate_out = self.conv_gate(X)[:, :, : X.size(2)]
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        residual_out = self.residual(out)
        skip_out = self.skip(out)
        return residual_out + X, skip_out


class WaveNet(nn.Module):
    def __init__(
        self, num_classes, in_channels, out_channels, kernel_size=2, num_blocks=3
    ):
        # super(WaveNet, self).__init__()
        super().__init__()
        self.num_channels = in_channels
        self.blocks = nn.ModuleList()
        self.num_blocks = num_blocks

        for i in range(num_blocks):
            self.blocks.append(
                WaveNetBlock(1, out_channels, kernel_size, dilation=2**i)
            )

        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, X):
        skip_connections = []

        for i in range(self.num_channels):
            x_i = X[:, i, :].unsqueeze(1)
            skip_i = 0

            for block in self.blocks:
                x_i, skip_out = block(x_i)
                skip_i += skip_out

            skip_connections.append(skip_i)

        X = torch.stack(skip_connections, dim=1).sum(dim=1)
        X = F.relu(self.final_conv(X))
        X = self.global_pool(X).squeeze(2)
        X = self.fc(X)
        return X


if __name__ == "__main__":
    from torchinfo import summary

    num_classes = 1854
    in_channels = 271
    seq_len = 281
    model = WaveNet(num_classes=num_classes, in_channels=in_channels, out_channels=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, (1, in_channels, seq_len), device=device.type)
