import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label



train_dir = 'data/train/'
test_dir = 'data/test/'

train_list = os.listdir(train_dir)
test_list = os.listdir(test_dir)

for i in range(len(train_list)):
    train_list[i] = os.path.join(train_dir, train_list[i])

for i in range(len(test_list)):
    test_list[i] = os.path.join(test_dir, test_list[i])

labels = [path.split('/')[-1].split('.')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=42)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=val_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=8, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=8, shuffle=True)

if __name__ == '__main__':
    print(len(train_data), len(train_loader))
    print(len(valid_data), len(valid_loader))
