import numpy as np
import torch
import torch.nn as nn


np.random.seed(777)
torch.manual_seed(777)


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def count_parameters(self, trainable=True):
        params = [
            param.numel()
            for param in self.parameters() if param.requires_grad == trainable
        ]
        return sum(params), params

    def __repr__(self):
        params = self.count_parameters()
        return f'{super().__repr__()}\ntrainable params: {params[0]:,d} {params[1]}'


class Net(BaseNet):
    def __init__(self, in_channels, num_fmaps=32):
        super().__init__()
        self.num_fmaps = num_fmaps

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_fmaps, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm2d(num_fmaps),
            nn.LeakyReLU(0.02)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_fmaps, num_fmaps * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_fmaps * 2),
            nn.LeakyReLU(0.02)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_fmaps * 2, num_fmaps * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_fmaps * 4),
            nn.LeakyReLU(0.02)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(num_fmaps * 4 * 4 * 4, 256),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc1(out.view(-1, self.num_fmaps * 4 * 4 * 4))
        out = self.fc2(out)

        return out




if __name__ == '__main__':
    model = Net(1, 32)
    print(model)
