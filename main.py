import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from dataloader1 import CustomDataloader
from model import VGG19_Net

transform =transforms.Compose([
    transforms.Resize((224,128)),
])
trainset=CustomDataloader('img',transform=transform,istrain=True)
testset=CustomDataloader('img',transform=transform,istrain=False)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
testloader=torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = VGG19_Net()
net = net.to(device)
param = list(net.parameters())

import torch.optim as optim

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(),lr=0.00001)

for epoch in range(3):  # loop over the dataset multiple times
    running_loss = 0.0

    if(epoch>0):
        net = VGG19_Net()
        net.load_state_dict(torch.load(save_path))
        net.to(device)

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs,f = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if(loss.item() > 1000):
            print(loss.item())
            for param in net.parameters():
                print(param.data)
        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
#########################################
    save_path="save/test/vgg19/vgg19_result.pth"
    torch.save(net.state_dict(), save_path)
    print('epoch %d finished'%(epoch+1))
###############################################
print('Finished Training')

class_correct = list(0. for i in range(136))
class_total = list(0. for i in range(136))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs,_ = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(64):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_sum=0
for i in range(136):
    temp = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %3d : %2d %%' % (
        i, temp))
    accuracy_sum+=temp
print('Accuracy average: ', accuracy_sum/136)