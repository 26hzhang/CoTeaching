import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = torch.argsort(loss_1.data).cuda()

    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    ind_2_sorted = torch.argsort(loss_2.data).cuda()

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * ind_1_sorted.size()[0])

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update.cpu().detach().numpy()]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update.cpu().detach().numpy()]]) / float(num_remember)

    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    return torch.sum(loss_1_update), torch.sum(loss_2_update), pure_ratio_1, pure_ratio_2


def call_bn(bn, x):
    return bn(x)


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit
