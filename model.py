import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch import nn
import torch.nn.functional as F
from torch import optim

root_path = "data/mask_data"
rotation_value = 30
crop_value = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define transforms
# train_transform = transforms.Compose([transforms.RandomRotation(rotation_value),
#                             transforms.RandomResizedCrop(crop_value), 
#                             transforms.RandomHorizontalFlip(), 
#                             transforms.ToTensor()])
train_transform = transforms.Compose([transforms.RandomRotation(rotation_value),
                            transforms.CenterCrop(224), 
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                            
test_transform = transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()])

# make datasets
train_dataset = datasets.ImageFolder(root_path, transform=train_transform)
test_dataset = datasets.ImageFolder(root_path, transform=test_transform)

# load data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

classes = ('nose', 'gamer')

dataiter = iter(train_loader)
images, labels = dataiter.next()

npimage = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(npimage, (1,2,0)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear (16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
print('Done training')
PATH = "./image_net.pth"
torch.save(net.state_dict(), PATH)