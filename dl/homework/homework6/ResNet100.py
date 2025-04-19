import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper function to count parameters and calculate FLOPs
def count_params_and_flops(model, input_size=(1, 3, 32, 32)):
    """Count parameters and estimate FLOPs using torchinfo."""
    model_summary = summary(model, input_size=input_size, verbose=0)
    params = model_summary.total_params
    flops = model_summary.total_mult_adds
    return params, flops, model_summary

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    pbar = tqdm(train_loader, desc='Training', leave=True)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar with current loss and accuracy
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    epoch_time = time.time() - start_time
    
    return train_loss, train_acc, epoch_time

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating', leave=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    
    return test_loss, test_acc

# Main function to run the ResNet experiment
def main():
    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)
    
    # Create ResNet-18 model
    resnet18 = torchvision.models.resnet18(weights=None)
    # Modify the first layer for CIFAR-100 (smaller images)
    resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18.maxpool = nn.Identity()  # Remove maxpool layer
    # Modify the final FC layer for 100 classes
    resnet18.fc = nn.Linear(512, 100)
    
    # Calculate parameters and FLOPs
    params, flops, model_summary = count_params_and_flops(resnet18)
    print(f"ResNet-18 Model Summary:")
    print(model_summary)
    print(f"Number of parameters: {params:,}")
    print(f"Estimated FLOPs per forward pass: {flops:,}")
    
    # Set up training
    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=0.001)
    
    # Training loop
    resnet18.to(device)
    results = {
        "train_time_per_epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, epoch_time = train(resnet18, trainloader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(resnet18, testloader, criterion, device)
        
        # Record results
        results["train_time_per_epoch"].append(epoch_time)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")
    
    # Calculate average metrics
    avg_train_time = np.mean(results["train_time_per_epoch"])
    final_test_acc = results["test_acc"][-1]
    
    print("\nFinal Results:")
    print(f"Average training time per epoch: {avg_train_time:.2f}s")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    
    # Visualize results
    epochs = range(1, num_epochs+1)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_acc"], label="Train Accuracy")
    plt.plot(epochs, results["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("resnet100_performance.png")
    plt.show()

if __name__ == "__main__":
    main() 