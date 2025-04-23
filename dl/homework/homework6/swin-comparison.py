import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification, SwinConfig, AutoImageProcessor
from tqdm import tqdm
import time
import pandas as pd
from copy import deepcopy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 3  # Using 3 epochs for fine-tuning (between 2-5 as requested)
batch_size = 32
learning_rate = 2e-5  # Smaller learning rate for fine-tuning
image_size = 224  # Swin expects 224x224 input by default

# Model configurations
models_config = {
    "swin-tiny-pretrained": {
        "name": "microsoft/swin-tiny-patch4-window7-224",
        "pretrained": True,
        "freeze_backbone": True
    },
    "swin-small-pretrained": {
        "name": "microsoft/swin-small-patch4-window7-224",
        "pretrained": True,
        "freeze_backbone": True
    },
    "swin-tiny-scratch": {
        "name": "microsoft/swin-tiny-patch4-window7-224",
        "pretrained": False,
        "freeze_backbone": False
    }
}

# Results tracking
results = {
    "model": [],
    "epoch_train_time": [],
    "test_accuracy": []
}

# CIFAR-100 dataset preparation
def prepare_data(model_name):
    # Data preparation with proper preprocessing for Swin
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Create and configure model
def setup_model(config):
    if config["pretrained"]:
        print(f"Loading pretrained {config['name']}...")
        model = SwinForImageClassification.from_pretrained(
            config["name"],
            num_labels=100,  # CIFAR-100 has 100 classes
            ignore_mismatched_sizes=True  # Allows replacing the original classifier head
        ).to(device)
    else:
        print(f"Initializing {config['name']} from scratch...")
        # For scratch training, initialize with the same architecture but random weights
        swin_config = SwinConfig.from_pretrained(
            config["name"],
            num_labels=100  # CIFAR-100 has 100 classes
        )
        model = SwinForImageClassification(swin_config).to(device)
    
    # Freeze backbone parameters if specified
    if config["freeze_backbone"]:
        print("Freezing backbone parameters...")
        for param in model.swin.parameters():
            param.requires_grad = False
        
        # Only the classifier head will be trained
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Configure optimizer for fine-tuning (only classifier parameters)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        print("Training all parameters...")
        # Configure optimizer for training from scratch (all parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer

# Training function
def train_model(model, optimizer, train_loader, test_loader, model_name):
    criterion = nn.CrossEntropyLoss()
    epoch_times = []
    
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            if (i+1) % 100 == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} training time: {epoch_time:.2f} seconds")
    
    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    # Test the model
    accuracy = test_model(model, test_loader)
    
    # Store results
    results["model"].append(model_name)
    results["epoch_train_time"].append(avg_epoch_time)
    results["test_accuracy"].append(accuracy)
    
    return avg_epoch_time, accuracy

# Testing function
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        return accuracy

# Main function
def main():
    for model_name, config in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Prepare data
        train_loader, test_loader = prepare_data(config["name"])
        
        # Setup model
        model, optimizer = setup_model(config)
        
        # Train and test model
        avg_epoch_time, accuracy = train_model(model, optimizer, train_loader, test_loader, model_name)
        
        print(f"Model: {model_name}")
        print(f"Average epoch training time: {avg_epoch_time:.2f} seconds")
        print(f"Test accuracy: {accuracy:.2f}%")
        
        # Clear GPU memory
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create and display results table
    results_df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv("swin_comparison_results.csv", index=False)
    print("Results saved to swin_comparison_results.csv")
    
    # Print findings for report
    print("\nKey Findings for Report:")
    print("1. Fine-tuning vs. Training from Scratch:")
    ft_acc = results_df[results_df['model'] == 'swin-tiny-pretrained']['test_accuracy'].values[0]
    scratch_acc = results_df[results_df['model'] == 'swin-tiny-scratch']['test_accuracy'].values[0]
    print(f"   - Accuracy difference: {ft_acc - scratch_acc:.2f}%")
    
    print("2. Swin-Tiny vs. Swin-Small:")
    tiny_acc = results_df[results_df['model'] == 'swin-tiny-pretrained']['test_accuracy'].values[0]
    small_acc = results_df[results_df['model'] == 'swin-small-pretrained']['test_accuracy'].values[0]
    print(f"   - Accuracy difference: {small_acc - tiny_acc:.2f}%")
    
    # Note about training times
    tiny_time = results_df[results_df['model'] == 'swin-tiny-pretrained']['epoch_train_time'].values[0]
    small_time = results_df[results_df['model'] == 'swin-small-pretrained']['epoch_train_time'].values[0]
    scratch_time = results_df[results_df['model'] == 'swin-tiny-scratch']['epoch_train_time'].values[0]
    print(f"3. Training Time Comparison:")
    print(f"   - Swin-Tiny (pretrained): {tiny_time:.2f} seconds/epoch")
    print(f"   - Swin-Small (pretrained): {small_time:.2f} seconds/epoch")
    print(f"   - Swin-Tiny (scratch): {scratch_time:.2f} seconds/epoch")

if __name__ == '__main__':
    main() 