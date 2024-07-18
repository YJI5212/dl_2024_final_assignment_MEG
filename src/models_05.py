import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.out_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_out = torch.cat([x, reset * h], dim=1)
        out = torch.tanh(self.out_gate(combined_out))
        new_h = (1 - update) * h + update * out
        return new_h


class BasicGRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len,
        in_channels,
        hidden_dim=128,
        num_layers=2,
        p_drop=0.1,
    ):
        super(BasicGRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru_cells = nn.ModuleList(
            [
                GRUCell(in_channels if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.rearrange = Rearrange("b d 1 -> b d")

    def forward(self, X):
        batch_size, _, seq_len = X.size()
        h = [
            torch.zeros(batch_size, self.hidden_dim, device=X.device)
            for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            x_t = X[:, :, t]
            for i, gru_cell in enumerate(self.gru_cells):
                h[i] = gru_cell(x_t, h[i])
                x_t = h[i]

        X = h[-1].unsqueeze(2)
        X = self.adaptive_avg_pool(X)
        X = self.rearrange(X)
        X = self.dropout(X)
        X = self.fc(X)
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
