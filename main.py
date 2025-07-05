import kagglehub
path = kagglehub.dataset_download("mohneesh7/english-alphabets")

print("Path to dataset files:", path)
print("Dataset laoding complete")

data_dir = 'C:/Users/User/.cache/kagglehub/datasets/mohneesh7/english-alphabets/versions/1/english_alphabets'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

class EnglishCharacterDataset(Dataset):
  def __init__(self, data_dir, transform = None):
    self.data = ImageFolder(data_dir, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]

  @property
  def classes(self):
    return self.data.classes
  
from random import randint

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = EnglishCharacterDataset(data_dir=data_dir, transform=transform)

# Test to see if dataset loads
print("Length of dataset: ", len(dataset))
image, label = dataset[randint(0, len(dataset) - 1)]
image

for image, label in dataset:
    break

# Making a dataloader in order to create batches to train
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

# Testing by iterating and breaking, we'll be using images as the testing in the model down below
for images, labels in dataloader:
  break

# Creating the classifier (untrained base model) to train from timm, the model class is called a 'Classifier' dunno why
class EnglishCharacterClassifier(nn.Module):
  def __init__(self, numClasses=26): # numClasses here is 26 because of 26 letters in the english alphabet
    super(EnglishCharacterClassifier, self).__init__()
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    enet_out_size = 1280
    # Make a classifier
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(enet_out_size, numClasses)
    )

  def forward(self, x):
    x = self.features(x)
    output = self.classifier(x)

    return output

# Creating a new model from the classifier above
model = EnglishCharacterClassifier(numClasses=26)
# print(model)

# Testing the model by passing one 'images' batch from the dataloader test from above
model(images) # Should output a tensor if works
output = model(images)
print(output.shape)

for images, labels in dataloader:
  continue

# TRAINING THE MODEL STARTS HERE
# First we need to make 2 things: a loss function (called a criterion) and an optimizer
model = EnglishCharacterClassifier(numClasses=26)
model(images)
output = model(images)
print(output.shape)

criterion = nn.CrossEntropyLoss();
optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion(output, labels)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

trainFolder = data_dir
validFolder = data_dir

trainDataset = EnglishCharacterDataset(data_dir = trainFolder, transform=transform)
validDataset = EnglishCharacterDataset(data_dir = validFolder, transform=transform)

trainDataLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
validDataLoader = DataLoader(validDataset, batch_size=32, shuffle=False)

# Simple training loop
num_epochs = 5
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = EnglishCharacterClassifier(numClasses=26)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should print: NVIDIA GeForce RTX 3050


for epoch in range(num_epochs):
  model.train() #!!!!!!!!!!!!!!!!!!!
  runningLoss = 0.0
  for images, labels in tqdm(trainDataLoader, desc='Training loop'):
    # Move inputs and labels to the device
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    runningLoss += loss.item() * labels.size(0)
  train_loss = runningLoss / len(trainDataLoader.dataset)
  train_losses.append(train_loss)

  # Validation phase
  model.eval()
  running_loss = 0.0

  with torch.no_grad():
    for images, labels in tqdm(validDataLoader, desc='Validation loop'):
      # Move inputs and labels to the device
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)
      running_loss += loss.item() * labels.size(0)

    val_loss = running_loss / len(validDataLoader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

