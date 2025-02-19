import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
print(device)


#import cifar10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Note: Data will be moved to GPU in training loop using:
# inputs, labels = inputs.to(device), labels.to(device)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = nn.Sequential(
    nn.Conv2d(3, 32, 3),  # Input channels: 3 (RGB), Output: 32 feature maps, 3x3 kernel
    nn.ReLU(),
    nn.MaxPool2d(2, 2),   # Reduce spatial dimensions by half
    nn.Conv2d(32, 64, 3), # 32 input channels, 64 output channels
    nn.ReLU(), 
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.Flatten(),         # Flatten for fully connected layers
    nn.Linear(64 * 6 * 6, 64),  # Calculate input size based on previous operations
    nn.ReLU(),
    nn.Linear(64, 10)     # 10 output classes for CIFAR10
)

model = model.to(device)  # Move model to GPU if available


print(model)