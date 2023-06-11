import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-2
epoch = 100
batch_size = 64

train_dataset = torchvision.datasets.CIFAR10("dataset", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=trans, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_dataset), len(test_dataset))

lenet = LeNet().to(device)
best_lenet, best_acc = LeNet().to(device), 0
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(lenet.parameters(), lr=learning_rate)

writer = SummaryWriter("logs_lenet5")

total_train_step = 0

for i in range(epoch):
    lenet.train()
    for data in train_dataloader:
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = lenet(imgs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss", loss, total_train_step)
            print(f"Step{total_train_step}:train_loss:{loss}")
        total_train_step += 1

    lenet.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = lenet(imgs)

            accuracy = (torch.argmax(outputs, 1) == labels).sum()
            total_accuracy += accuracy

    torch.save(lenet.state_dict(), f"model/Trained_Lenet{i}.pth")
    writer.add_scalar("test_acc", total_accuracy / len(test_dataset), i)
    print(f"Epoch{i} acc:{total_accuracy / len(test_dataset)}")

writer.close()
# torch.save(best_lenet.state_dict(), "Trained_Lenet5.pth")