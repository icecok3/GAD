import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from MNIST import MNIST, test
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attacker():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def FGSM(self, X, y, is_target=False, eps=0.5):
        x_adv = X
        x_adv.to(device)
        x_adv.requires_grad = True

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
            eta = torch.clamp(x_adv - x_org, -eps, eps)
            x_adv = torch.clamp(x_org + eta, 0, 1).detach_()

        return x_adv

if __name__ == "__main__":
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    testdata = torchvision.datasets.MNIST(root="../data", train=False, transform=mnist_transform)
    testloader = DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

    model = MNIST()
    model.load_state_dict(torch.load('../saved_models/MNIST.pt'))
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    attacker = Attacker(model, criterion)
    for eps in range(0, 6):
        acc_tol = 0
        eps = eps/10

        for step, (X, y) in enumerate(testloader):
            X, y = X.to(device), y.to(device)
            x_adv = attacker.PGD(X, y, eps=eps)
            output, loss, acc = test(model, x_adv, y, criterion)
            acc_tol += acc
            output = torch.max(output, 1)[1]
            # acc = torch.sum(out == y) / len(X)
            # acc = acc/len(X)
        acc = torch.true_divide(acc_tol, len(testloader.dataset))
        print(acc.item())

        res_plot(x_adv.squeeze()[:9], y[:9].cpu().numpy(), output[:9].cpu().detach().numpy(), 3, 3, "eps:{}   acc:{}".format(eps, acc))







