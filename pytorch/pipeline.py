import torch
import torch.nn as nn
import torch.nn.functional as F



class neuralnet(nn.Module):
  def __init__(self,num_classes=2):
    super().__init__()
    self.conv1=nn.Conv2d(3,12,5)
    self.pool =nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(12,24,5)

    self.adaptive_ppol = nn.AdaptiveAvgPool2d((6,6))


    self.fc1=nn.Linear(24*6*6,120)
    self.fc2=nn.Linear(120,84)
    self.fc3=nn.Linear(84,16)
    self.fc4 =nn.Linear(16,num_classes)

  def forward(self,x):
    x=self.pool(F.relu(self.conv1(x)))
    x=self.pool(F.relu(self.conv2(x)))
    x=self.adaptive_ppol(x)
    x=torch.flatten(x,1)
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=F.relu(self.fc3(x))
    x=self.fc4(x)
    return x