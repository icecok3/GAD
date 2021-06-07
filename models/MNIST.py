import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.conv1 = nn.Conv2d(1, 10, 5)
        # self.conv2 = nn.Conv2d(10, 20, 5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

def accuracy(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, 1) == y_true).data

def train(model, X, y, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, y)
    acc = torch.sum(torch.argmax(output, 1) == y).data

    loss.backward()
    optimizer.step()

    return output, loss, acc

@torch.no_grad()
def test(model, X, y, criterion):
    model.eval()

    output = model(X)
    loss = criterion(output, y)
    acc = torch.sum(torch.argmax(output, 1) == y).data

    return output, loss, acc

def MNIST_data():
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    traindata = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=mnist_transform)
    testdata = torchvision.datasets.MNIST(root="../data", train=False, transform=mnist_transform)
    trainloader = DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
    testloader = DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

    return traindata, testdata, trainloader, testloader

def train_MNIST(model, epochs, criterion, optimizer, save_path = None):
    model.to(device)

    traindata, testdata, trainloader, testloader = MNIST_data()
    print("shape of train dataset: {}\nshape of test dataset: {}".format(tuple(traindata.data.shape), tuple(testdata.data.shape)))

    for epoch in range(1, epochs + 1):
        loss_sum = 0
        acc_num = 0
        for step, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)

            pred, loss, acc = train(model, X, y, criterion, optimizer)
            loss_sum += loss
            acc_num += acc

        val_loss_sum = 0
        val_acc_num = 0
        for val_step, (X_val, y_val) in enumerate(testloader):
            X_val, y_val = X_val.cuda(), y_val.cuda()

            output, val_loss, val_acc = test(model, X_val, y_val, criterion)
            val_loss_sum += val_loss
            val_acc_num += val_acc

        if epoch % 1 == 0:
            print("="*50)
            print("Epoch:{}, loss:{}, acc:{}, val_loss:{}, val_acc:{}".format(epoch,
                                                                              loss_sum / len(trainloader.dataset),
                                                                              acc_num / len(trainloader.dataset),
                                                                              val_loss_sum / len(testloader.dataset),
                                                                              val_acc_num / len(testloader.dataset)
                                                                              ))
            print("="*50)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    model = MNIST()
    # summary(model, (1, 28, 28), device='cpu')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_MNIST(model, 10, criterion=criterion, optimizer=optimizer, save_path = '../saved_models/MNIST.pt')


