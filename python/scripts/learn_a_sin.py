import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module


class SinNet(Module):

    def __init__(self):
        super(SinNet, self).__init__()

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


    sin_net = SinNet()

    x_tr = Variable(torch.rand(1, 1)).cuda()
    x_val = Variable(torch.rand(500, 1), requires_grad=False).cuda()
    x_tr.view()
    C = 10
    y_val = torch.sin(torch.mul(x_val, C))

    criterion_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(sin_net.parameters(), lr=1e-4)

    for i in range(50):
        y_tr = torch.sin(torch.mul(x_tr, C))
        y_pred = sin_net(x_tr)

        tr_loss = criterion_fn(y_pred, y_tr)

        y_pred_val = sin_net(x_val)
        val_loss = criterion_fn(y_pred_val, y_val)
        
        print("Losses: {0:.7f} {0:.7f}".format(tr_loss.data[0], val_loss.data[0]))

        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

    x_pred = Variable(torch.mul(torch.rand(500, 1), C), requires_grad=False).cuda()
    y_pred = sin_net(x_pred)


