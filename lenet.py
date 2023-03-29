import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 3,224,224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)    # 6,220,220
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      # 6,110,110
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)   # 16,106,106
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                      # 16,53,53

        self.fc1 = nn.Linear(in_features=16 * 53 * 53, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=2)

    def forward(self, x):     
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




if __name__ == "__main__":
    net = LeNet()
    print(net)