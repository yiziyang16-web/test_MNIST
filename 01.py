import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 自己搭建网络
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练函数
def train(model, loader, crit, opt, dev):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for img, label in loader:
        img, label = img.to(dev), label.to(dev)
        opt.zero_grad()
        out = model(img)
        loss = crit(out, label)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# 测试函数
def test(model, loader, crit, dev):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(dev), label.to(dev)
            out = model(img)
            loss = crit(out, label)
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# 开始跑
epochs = 5
for ep in range(epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"Epoch [{ep+1}/{epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f},  Test  Acc: {test_acc:.4f}\n")
