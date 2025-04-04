import os 
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize(255),
                                 transforms.ToTensor()])

dataset = datasets.ImageFolder('Dataset', transform=transform)

print(len(dataset))
print(dataset.classes)
