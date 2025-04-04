import os 
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder('Dataset', transform=transform)
num_classes = len(dataset.classes)

train_dataset, validation_dataset = train_test_split(dataset,test_size=0.2,train_size=0.8,shuffle=True,random_state=42,stratify=[tp[1] for tp in dataset])
test_dataset, validation_dataset = train_test_split(validation_dataset,test_size=0.5,train_size=0.5,shuffle=True,random_state=42,stratify=[tp[1] for tp in validation_dataset])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

print(dict(Counter([label for _, label in dataset])))
print(dict(Counter([label for _, label in train_dataset])))
print(dict(Counter([label for _, label in validation_dataset])))
print(dict(Counter([label for _, label in test_dataset])))






# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze feature extractor layers (optional)
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for your dataset
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")