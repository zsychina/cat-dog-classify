import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt

train_root = r'./data/train'
test_root = r'./data/test/test'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()          
])

train_dataset = ImageFolder(
    train_root,
    transform=transform
)

train_size = len(train_dataset)
test_size = int(train_size * 0.2)
train, test = random_split(train_dataset, [train_size-test_size, test_size])

train_loader = DataLoader(
    train,
    batch_size=4,
    shuffle=True,
)

test_loader = DataLoader(
    test,
    batch_size=4,
    shuffle=True,
)


if __name__ == '__main__':
    print(train_dataset)
    print(len(train_dataset))
    # print(test_dataset)
    # print(len(test_dataset))

    # plt.imshow(train_dataset[1010][0].permute(1,2,0))
    # plt.show()
