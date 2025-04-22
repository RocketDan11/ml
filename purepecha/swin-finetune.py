import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import ast
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 5
batch_size = 16
learning_rate = 2e-5
max_length = 128
warmup_steps = 100

# Load Purepecha data
def load_purepecha_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the data list from the file
        data_str = content.strip()
        if data_str.startswith('data = '):
            data_str = data_str[7:]
        data = ast.literal_eval(data_str)
    return data

# Create a custom dataset for translation
class PurepechaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        eng, pure = self.data[idx]
        
        # Tokenize the input and target
        source_encoding = self.tokenizer(
            eng,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            pure,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# Training function
def train(model, train_loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    return avg_loss

# Evaluation function
def evaluate(model, eval_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_loader)
    print(f'Evaluation Loss: {avg_loss:.4f}')
    return avg_loss

# Translation function
def translate(model, tokenizer, text, max_length=128):
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode the generated ids to text
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def main():
    # Load the Purepecha data
    data = load_purepecha_data('purepecha/assets/purepecha_data.txt')
    print(f"Loaded {len(data)} translation pairs")
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    print(f"Training set: {len(train_data)} pairs, Validation set: {len(val_data)} pairs")
    
    # Load pre-trained model and tokenizer
    # Using Helsinki-NLP's MarianMT model for English to Spanish as a starting point
    # We'll fine-tune it for English to Purepecha
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    
    # Create datasets
    train_dataset = PurepechaDataset(train_data, tokenizer, max_length)
    val_dataset = PurepechaDataset(val_data, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, epoch)
        val_loss = evaluate(model, val_loader)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'purepecha_best_model.pt')
            print(f"Model saved with validation loss: {val_loss:.4f}")
    
    # Load the best model for translation
    model.load_state_dict(torch.load('purepecha_best_model.pt'))
    
    # Test translations
    test_phrases = [
        "Hello, how are you?",
        "I love you",
        "What is your name?",
        "I am learning Purepecha"
    ]
    
    print("\nTest Translations:")
    for phrase in test_phrases:
        translation = translate(model, tokenizer, phrase)
        print(f"English: {phrase}")
        print(f"Purepecha: {translation}")
        print()

if __name__ == '__main__':
    main() 