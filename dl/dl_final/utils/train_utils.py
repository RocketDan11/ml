"""
Training and evaluation utilities for translation models.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import random
import inspect

class Trainer:
    """
    Trainer class for sequence-to-sequence models.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        clip_grad=1.0,
        checkpoint_dir="checkpoints",
        teacher_forcing_ratio=0.5,
        early_stopping_patience=5
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.clip_grad = clip_grad
        self.checkpoint_dir = checkpoint_dir
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.early_stopping_patience = early_stopping_patience
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        self.val_bleu_scores = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            src = batch["source"].to(self.device)
            src_lengths = batch["source_lengths"].to(self.device)
            trg = batch["target"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio
            
            # Forward pass - handle different model parameter names
            if hasattr(self.model, 'forward') and 'teacher_forcing_ratio' in inspect.signature(self.model.forward).parameters:
                output = self.model(src, src_lengths, trg, teacher_forcing_ratio=self.teacher_forcing_ratio if use_teacher_forcing else 0.0)
            else:
                output = self.model(src, src_lengths, trg, use_teacher_forcing=use_teacher_forcing)
            
            # Handle case where output is a tuple (outputs, attention)
            if isinstance(output, tuple):
                output = output[0]  # Extract just the output tensor, not the attention weights
            
            # Ignore padding tokens in the target
            # Output shape: [batch_size, trg_len - 1, output_dim]
            # Target shape: [batch_size, trg_len]
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)  # Exclude <SOS> token
            
            # Calculate loss
            loss = self.criterion(output, trg)
            
            # Backward pass and optimization
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return epoch_loss / len(self.train_loader)
    
    def evaluate(self, calculate_bleu=True):
        """
        Evaluate the model on the validation set.
        
        Args:
            calculate_bleu: Whether to calculate BLEU score
            
        Returns:
            Tuple of (validation loss, BLEU score)
        """
        self.model.eval()
        epoch_loss = 0
        
        # Lists for storing target and predicted sequences for BLEU calculation
        all_trg_tokens = []
        all_pred_tokens = []
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.val_loader, desc="Validating")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                src = batch["source"].to(self.device)
                src_lengths = batch["source_lengths"].to(self.device)
                trg = batch["target"].to(self.device)
                
                # Forward pass (without teacher forcing)
                if hasattr(self.model, 'forward') and 'teacher_forcing_ratio' in inspect.signature(self.model.forward).parameters:
                    output = self.model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
                else:
                    output = self.model(src, src_lengths, trg, use_teacher_forcing=False)
                
                # Handle case where output is a tuple (outputs, attention)
                if isinstance(output, tuple):
                    output = output[0]  # Extract just the output tensor, not the attention weights
                
                # Ignore padding tokens in the target
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)  # Exclude <SOS> token
                
                # Calculate loss
                loss = self.criterion(output, trg)
                
                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Store target and predicted tokens for BLEU calculation
                if calculate_bleu:
                    # Get target tokens (exclude <SOS> and <PAD>)
                    for i in range(batch["target"].shape[0]):
                        trg_len = batch["target_lengths"][i] - 1  # -1 for <SOS>
                        trg_tokens = batch["target"][i][1:trg_len].tolist()  # exclude <SOS>
                        trg_tokens = [batch["target_text"][i]]  # Use actual text instead of tokens
                        all_trg_tokens.append(trg_tokens)
                        
                    # Get predicted tokens
                    batch_size = batch["target"].shape[0]
                    trg_len = output.shape[0] // batch_size
                    output_dim = output.shape[-1]
                    preds = output.view(batch_size, trg_len, output_dim).argmax(dim=2)
                    
                    for i in range(batch_size):
                        # Decode predicted tokens until <EOS> or end of sequence
                        pred_tokens = []
                        for j in range(trg_len):
                            token_idx = preds[i, j].item()
                            if token_idx == 2:  # <EOS> token
                                break
                            if token_idx not in [0, 1, 2, 3]:  # Exclude special tokens
                                pred_tokens.append(self.model.decoder.vocab.itos[token_idx])
                                
                        all_pred_tokens.append([" ".join(pred_tokens)])
        
        val_loss = epoch_loss / len(self.val_loader)
        
        # Calculate BLEU score
        bleu_score = 0
        if calculate_bleu and all_trg_tokens and all_pred_tokens:
            # Option 1: Use corpus_bleu with smoothing
            smoothie = SmoothingFunction().method3
            
            # Option 2: Use sentence_bleu with averaging
            bleu_scores = []
            for i in range(len(all_trg_tokens)):
                reference = [all_trg_tokens[i][0].split()]
                hypothesis = all_pred_tokens[i][0].split()
                # Calculate sentence-level BLEU with smoothing
                sentence_bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
                bleu_scores.append(sentence_bleu_score)
            
            # Average the sentence-level BLEU scores
            bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            
        return val_loss, bleu_score
    
    def train(self, epochs):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary of training metrics
        """
        print(f"Training model for {epochs} epochs on {self.device}")
        
        # Record start time
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluate on validation set
            val_loss, bleu_score = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_bleu_scores.append(bleu_score)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}")
            
            # Save checkpoint if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save model checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model.__class__.__name__}_best.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bleu_score': bleu_score
                }, checkpoint_path)
                
                print(f"Saved checkpoint to {checkpoint_path}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
        # Calculate total training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model
        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model.__class__.__name__}_best.pt")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_bleu_scores': self.val_bleu_scores,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation BLEU score
        plt.subplot(2, 1, 2)
        plt.plot(self.val_bleu_scores, label='Validation BLEU Score', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('Validation BLEU Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

def translate_sentence(model, sentence, source_vocab, target_vocab, device, max_length=50):
    """
    Translate a sentence using the model.
    
    Args:
        model: Trained translation model
        sentence: Source sentence to translate
        source_vocab: Source vocabulary
        target_vocab: Target vocabulary
        device: Device to run inference on
        max_length: Maximum length of generated translation
        
    Returns:
        Translated sentence
    """
    model.eval()
    
    # Tokenize and numericalize the sentence
    tokens = source_vocab.tokenize(sentence)
    indices = [source_vocab.stoi.get(token, source_vocab.stoi["<UNK>"]) for token in tokens]
    
    # Add <SOS> and <EOS> tokens
    indices = [source_vocab.stoi["<SOS>"]] + indices + [source_vocab.stoi["<EOS>"]]
    source_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    source_length = torch.tensor([len(indices)]).to(device)
    
    with torch.no_grad():
        # Generate translation
        outputs = model.translate(source_tensor, source_length, max_length)
    
    # Convert output indices to tokens
    translated_tokens = []
    for idx in outputs:
        token = target_vocab.itos[idx]
        if token == "<EOS>":
            break
        if token not in ["<SOS>", "<PAD>", "<UNK>"]:
            translated_tokens.append(token)
    
    return " ".join(translated_tokens)

def evaluate_bleu(model, test_loader, target_vocab, device):
    """
    Evaluate the model's BLEU score on the test set.
    
    Args:
        model: Trained translation model
        test_loader: DataLoader for test set
        target_vocab: Target vocabulary
        device: Device to run inference on
        
    Returns:
        BLEU score
    """
    model.eval()
    references = []
    hypotheses = []
    bleu_scores = []
    smoothie = SmoothingFunction().method3
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating BLEU"):
            src = batch["source"].to(device)
            src_lengths = batch["source_lengths"].to(device)
            
            # Generate translations
            batch_translations = model.translate_batch(src, src_lengths)
            
            # Process each translation in the batch
            for i, translation in enumerate(batch_translations):
                # Get reference translation (actual target)
                reference = batch["target_text"][i].split()
                references.append([reference])
                
                # Get hypothesis (predicted translation)
                hypothesis = []
                for idx in translation:
                    token = target_vocab.itos[idx]
                    if token == "<EOS>":
                        break
                    if token not in ["<SOS>", "<PAD>", "<UNK>"]:
                        hypothesis.append(token)
                
                hypotheses.append(hypothesis)
                
                # Calculate sentence-level BLEU with smoothing
                sentence_bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
                bleu_scores.append(sentence_bleu_score)
    
    # Option 1: Calculate corpus-level BLEU with smoothing
    corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    
    # Option 2: Average the sentence-level BLEU scores
    avg_sentence_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # Return the average of sentence-level BLEU scores to be consistent with the evaluation method
    return avg_sentence_bleu