import torch
from lenet import LeNet
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using '+str(device)+"!")

model = LeNet()
model = model.to(device)
weights = torch.load('./target/temp_new.pt')
model.load_state_dict(weights)

root = r'./data/test/'
imgnames = os.listdir(root)
imgname = random.choice(imgnames)
path = os.path.join(root + imgname)
img = Image.open(path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()          
])

img_tensor = transform(img)

img_tensor = img_tensor.to(device)
# print(img_tensor.shape)


result = model(img_tensor)
result = F.softmax(result, dim=1)
class_label = torch.argmax(result)
if class_label == 0:
    t = 'cat'
    print('cat')
elif class_label == 1:
    t = 'dog'
    print('dog')

plt.imshow(img)
plt.text(0, 0, t, fontsize=20)
plt.show()