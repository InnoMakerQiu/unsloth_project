import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import Trainer,TrainingArguments
from sklearn.metrics import accuracy_score


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

# 将模型放入CUDA设备中
model = BasicNet()

training_args = TrainingArguments(
    "basic-trainer",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    remove_unused_columns=False,
#    deepspeed="./deepspeed_config.json"
)

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x":pixel_values, "labels":labels}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["x"])
        target = inputs["labels"]
        loss = F.nll_loss(outputs, target)
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(logits, dim=-1)
        accuracy = accuracy_score(labels, preds)
        return {"accuracy": accuracy}

trainer = MyTrainer(
    model,
    training_args,
    train_dataset=train_dset,
    eval_dataset=test_dset,
    data_collator=collate_fn,
)

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")

