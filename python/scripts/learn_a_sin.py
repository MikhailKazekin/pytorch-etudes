import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module


class FuncNet(Module):
    def __init__(self):
        super(FuncNet, self).__init__()

        H = 20
        self._layers = [
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1),
        ]

        self._model = torch.nn.Sequential(*self._layers)

        self._model.cuda()

    def forward(self, *input):
        return self._model(*input)


if __name__ == '__main__':

    func_net = FuncNet()

    loss_fn = torch.nn.MSELoss(size_average=False)
    loss_val_fn = torch.nn.MSELoss(size_average=False)

    optimizer = torch.optim.SGD(func_net.parameters(), lr=1e-4)

    C = 100
    batch_size = 32
    val_sample_size = 100

    f = lambda x: torch.sqrt(x)

    for epoch in range(50000):
        sample = torch.mul(torch.rand(batch_size, 1), C)
        val_sample = torch.mul(torch.rand(val_sample_size, 1), C)

        x = Variable(sample).cuda()

        y_pred = func_net(x)
        y = f(x)

        tr_loss = loss_fn(y_pred, y)

        x_val = Variable(val_sample, requires_grad=False).cuda()
        y_val = f(x_val)
        y_val_pred = func_net(x_val)
        val_loss = loss_val_fn(y_val_pred, y_val)

        if epoch % 2000 == 0:
            print(
            "({2}) Losses: {0:.5f} {1:.5f}".format(tr_loss.data.cpu().numpy()[0], val_loss.data.cpu().numpy()[0], epoch))

        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()