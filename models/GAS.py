import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import MNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAS():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

class Attacker():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def FGSM(self, X, y, is_target=False, eps=0.5):
        x_adv = X
        x_adv.to(device)
        x_adv.requires_grad = True

        print(next(model.parameters()).device, x_adv.device)

        output = self.model(x_adv)
        yy = y.add(2) % 10
        loss = self.criterion(output, yy)

        self.model.zero_grad()
        loss.backward()
        x_grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv - eps * x_grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

    def PGD(self, X, y, iters=50, is_target=False, eps=0.5):
        x_adv = X
        x_org = X
        x_adv.to(device)
        x_org.to(device)

        for i in range(iters):
            x_adv.requires_grad = True
            output = self.model(x_adv)
            yy = y.add(2) % 10
            loss = self.criterion(output, yy)

            self.model.zero_grad()
            loss.backward()
            x_grad_sign = x_adv.grad.data.sign()
            x_adv = x_adv - eps * x_grad_sign
            xx = torch.clamp(x_adv - x_org, -eps, eps)
            x_adv = torch.clamp(x_org + xx, 0, 1).detach_()

        return x_adv

        # X, y = X.to(device), y.to(device)
        # x_adv = X
        # x_org = X.data
        # x_adv.requires_grad = True
        # for i in range(iters):
        #     output = self.model(x_adv)
        #     loss = self.criterion(output, y).to(device)
        #     self.model.zero_grad()
        #     loss.backward()
        #
        #     x_grad_sign = x_adv.grad.data.sign()
        #     x_adv = (x_adv + eps * x_grad_sign).retain_grad()
        #     # x_adv = torch.clamp(x_adv, 0, 1)
        #     x_adv = torch.clamp(x_adv-x_org, -eps, eps)
        #     x_adv = torch.clamp(x_org + x_adv, 0, 1).detach_()
        #
        #
        # return x_adv


class FGSM(GAS):
    def __init__(self, model, criterion):
        super(FGSM, self).__init__(model, criterion)

    def attack(self, X, y, is_target=False, eps=0.5):

        x_adv = X
        x_adv.to(device)
        x_adv.requires_grad = True

        output = self.model(x_adv)
        yy = y.add(2) % 10
        loss = self.criterion(output, yy)

        self.model.zero_grad()
        loss.backward()
        x_grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv - eps*x_grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv


if __name__ == "__main__":
    eps = 0.2

    mnist_transform = transforms.Compose([transforms.ToTensor()])
    testdata = torchvision.datasets.MNIST(root="../data", train=False, transform=mnist_transform)
    testloader = DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

    model = MNIST.MNIST()
    model.load_state_dict(torch.load('../saved_models/MNIST.pt'))
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # attacker = FGSM(model, criterion)
    attacker = Attacker(model, criterion)
    for eps in range(0, 10):
        eps = eps/10
        # X, y = torch.Tensor(testdata.data.float().unsqueeze(1)), torch.Tensor(testdata.targets.float().unsqueeze(1))
        # X, y = (testdata.data.float().unsqueeze(1)), np.squeeze(testdata.targets)
        X, y = next(iter(testloader))
        X, y = X.to(device), y.to(device)
        # x_adv = attacker.attack(X, y, eps=eps)
        x_adv = attacker.PGD(X, y, eps=eps)
        output, loss, acc = MNIST.test(model, x_adv, y, criterion)
        output = torch.max(output, 1)[1]
        # acc = torch.sum(out == y) / len(X)
        acc = acc/len(X)
        print(acc.item())

        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(10, 10))
        axes = axes.flatten()
        for i in range(len(axes)):
            axes[i].imshow(x_adv.squeeze()[i].cpu().data)
            axes[i].set_title("{}->{}".format(y[i].cpu().numpy(), output[i].cpu().detach().numpy()))

        # axes[0].imshow(X.squeeze()[0].data)
        # axes[0].set_title(y[0].numpy())
        # axes[1].imshow(x_adv.squeeze()[0].data)
        # axes[1].set_title(out[0].detach().numpy())
        plt.suptitle("eps:{}   acc:{}".format(eps, acc))
        plt.show()






