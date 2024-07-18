import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ハイパーパラメータの設定
batch_size = 64
num_classes = 1854
num_epochs = 80
learning_rate = 1e-4
data_dir = "data/Images"

# データ拡張と正規化の設定
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# データセットの読み込み
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

# モデルのロード
efficientnet = timm.create_model(
    "efficientnet_b0", pretrained=True, num_classes=num_classes
)
regnety = timm.create_model("regnety_008", pretrained=True, num_classes=num_classes)

# 最後の全結合層を置き換え
efficientnet.classifier = nn.Linear(efficientnet.num_features, num_classes)
regnety.head.fc = nn.Linear(regnety.num_features, num_classes)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet = efficientnet.to(device)
regnety = regnety.to(device)

# ロス関数とオプティマイザの設定
criterion = nn.CrossEntropyLoss()
optimizer_efficientnet = optim.Adam(efficientnet.parameters(), lr=learning_rate)
optimizer_regnety = optim.Adam(regnety.parameters(), lr=learning_rate)

# モデルを保存するディレクトリを指定
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# ベストモデルを追跡するための変数
best_acc_efficientnet = 0.0
best_acc_regnety = 0.0

# トレーニングループ
for epoch in range(num_epochs):
    efficientnet.train()
    regnety.train()

    running_loss_efficientnet = 0.0
    running_loss_regnety = 0.0
    correct_efficientnet = 0
    correct_regnety = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_efficientnet.zero_grad()
        optimizer_regnety.zero_grad()

        outputs_efficientnet = efficientnet(inputs)
        outputs_regnety = regnety(inputs)

        loss_efficientnet = criterion(outputs_efficientnet, labels)
        loss_regnety = criterion(outputs_regnety, labels)

        loss_efficientnet.backward()
        optimizer_efficientnet.step()

        loss_regnety.backward()
        optimizer_regnety.step()

        running_loss_efficientnet += loss_efficientnet.item()
        running_loss_regnety += loss_regnety.item()

        _, predicted_efficientnet = torch.max(outputs_efficientnet, 1)
        _, predicted_regnety = torch.max(outputs_regnety, 1)

        total += labels.size(0)
        correct_efficientnet += (predicted_efficientnet == labels).sum().item()
        correct_regnety += (predicted_regnety == labels).sum().item()

    epoch_acc_efficientnet = correct_efficientnet / total
    epoch_acc_regnety = correct_regnety / total

    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"EfficientNet Loss: {running_loss_efficientnet/len(train_loader):.4f}, "
        f"EfficientNet Accuracy: {correct_efficientnet/total:.4f}, "
        f"RegNetY Loss: {running_loss_regnety/len(train_loader):.4f}, "
        f"RegNetY Accuracy: {correct_regnety/total:.4f}"
    )

    # ベストモデルの保存
    if epoch_acc_efficientnet > best_acc_efficientnet:
        best_acc_efficientnet = epoch_acc_efficientnet
        torch.save(
            efficientnet.state_dict(), os.path.join(save_dir, "efficientnet_best.pth")
        )
        print("Saved best EfficientNet model")

    if epoch_acc_regnety > best_acc_regnety:
        best_acc_regnety = epoch_acc_regnety
        torch.save(regnety.state_dict(), os.path.join(save_dir, "regnety_best.pth"))
        print("Saved best RegNetY model")

print("Finished Training")
