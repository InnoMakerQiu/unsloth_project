import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import Trainer,TrainingArguments
import accelerate
from accelerate import Accelerator

# 下面定义一个基本网络块
class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x,labels=None):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        if labels is not None:
            loss = F.nll_loss(output, labels)
            return loss, output
        else:
            return output

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

model = BasicNet()
accelerator = Accelerator()
device = accelerator.device
model.to(device)

# Build optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Send everything through `accelerator.prepare`
train_loader, test_loader, model, optimizer = accelerator.prepare(
    train_loader, test_loader, model, optimizer
)

# Train for a single epoch
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


# 在评估模式下禁用梯度计算
total_correct = 0
total_samples = 0
# Evaluate
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        # 使用 accelerator.gather_for_metrics 收集所有设备上的预测和目标
        all_predictions, all_targets = accelerator.gather_for_metrics((pred, target.view_as(pred)))
        # 将收集到的预测和目标添加到指标计算中
        correct = all_predictions.eq(all_targets).sum().item()
        if accelerator.is_main_process:
            # 更新总正确数和总样本数
            total_correct += correct
            total_samples += all_targets.size(0)
        


# 计算整体准确率
if accelerator.is_main_process:
    accuracy = 100. * total_correct/ total_samples
    print(f"Accuracy: {accuracy:.2f}%")



# # 使用 accelerator.print 打印结果
# accelerator.print(f"Accuracy: {final_score['accuracy']:.2f}%")
