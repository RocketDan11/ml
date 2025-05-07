"""
Main script for Purépecha-English translation using neural network models.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

# Import project modules
import config
from utils.data_utils import load_data
from utils.train_utils import Trainer
from utils.metrics import compare_models, plot_comparison, visualize_attention
from models.rnn_model import RNNSeq2Seq
from models.attention_model import AttentionSeq2Seq
from models.transformer_model import TransformerSeq2Seq

# Import LLM model conditionally
try:
    from models.llm_model import LLMTranslationModel, fine_tune_llm
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: LLM dependencies not available. LLM model will be disabled.")
    LLM_AVAILABLE = False

def train_rnn_model(data_loaders, device, use_gru=False):
    """
    Train LSTM/GRU RNN model.
    
    Args:
        data_loaders: Dictionary of data loaders and vocabularies
        device: Device to run training on
        use_gru: Whether to use GRU instead of LSTM
        
    Returns:
        Trained model and training metrics
    """
    print(f"\n{'='*20} Training {'' if not use_gru else 'GRU'} RNN Model {'='*20}")
    
    # Create model
    model = RNNSeq2Seq(
        encoder_vocab_size=len(data_loaders["source_vocab"]),
        decoder_vocab_size=len(data_loaders["target_vocab"]),
        encoder_embedding_dim=config.EMBEDDING_DIM,
        decoder_embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        encoder_dropout=config.DROPOUT,
        decoder_dropout=config.DROPOUT,
        rnn_type='gru' if use_gru else 'lstm',
        device=device
    ).to(device)
    
    # Set decoder vocabulary
    model.decoder.vocab = data_loaders["target_vocab"]
    
    # Skip model summary for now to avoid errors
    print(f"Model created: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train_loader"],
        val_loader=data_loaders["val_loader"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        clip_grad=config.CLIP_GRAD,
        checkpoint_dir=os.path.join(config.CHECKPOINT_DIR, f"{data_loaders['direction']}_rnn_gru" if use_gru else f"{data_loaders['direction']}_rnn_lstm"),
        teacher_forcing_ratio=config.TEACHER_FORCING_RATIO,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Train model
    metrics = trainer.train(config.EPOCHS)
    
    # Plot metrics
    model_name = 'GRU RNN' if use_gru else 'LSTM RNN'
    save_path = f"results/{data_loaders['direction']}_{model_name.lower().replace(' ', '_')}_metrics.png"
    trainer.plot_metrics(save_path)
    
    return model, metrics

def train_attention_model(data_loaders, device, use_gru=False):
    """
    Train RNN with attention model.
    
    Args:
        data_loaders: Dictionary of data loaders and vocabularies
        device: Device to run training on
        use_gru: Whether to use GRU instead of LSTM
        
    Returns:
        Trained model and training metrics
    """
    print(f"\n{'='*20} Training {'' if not use_gru else 'GRU'} RNN with Attention Model {'='*20}")
    
    # Create model
    model = AttentionSeq2Seq(
        encoder_vocab_size=len(data_loaders["source_vocab"]),
        decoder_vocab_size=len(data_loaders["target_vocab"]),
        encoder_embedding_dim=config.EMBEDDING_DIM,
        decoder_embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        num_layers=config.NUM_LAYERS,
        encoder_dropout=config.DROPOUT,
        decoder_dropout=config.DROPOUT,
        rnn_type='gru' if use_gru else 'lstm',
        device=device
    ).to(device)
    
    # Set decoder vocabulary
    model.decoder.vocab = data_loaders["target_vocab"]
    
    # Skip model summary for now to avoid errors
    print(f"Model created: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train_loader"],
        val_loader=data_loaders["val_loader"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        clip_grad=config.CLIP_GRAD,
        checkpoint_dir=os.path.join(config.CHECKPOINT_DIR, f"{data_loaders['direction']}_attention_gru" if use_gru else f"{data_loaders['direction']}_attention_lstm"),
        teacher_forcing_ratio=config.TEACHER_FORCING_RATIO,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Train model
    metrics = trainer.train(config.EPOCHS)
    
    # Plot metrics
    model_name = 'GRU with Attention' if use_gru else 'LSTM with Attention'
    save_path = f"results/{data_loaders['direction']}_{model_name.lower().replace(' ', '_')}_metrics.png"
    trainer.plot_metrics(save_path)
    
    # Visualize attention for a sample
    sample_idx = 0
    sample = data_loaders["test_dataset"][sample_idx]
    source_text = sample["source_text"]
    
    attention_viz = visualize_attention(
        model,
        source_text,
        data_loaders["source_vocab"],
        data_loaders["target_vocab"],
        device,
        save_path=f"results/{data_loaders['direction']}_{model_name.lower().replace(' ', '_')}_attention.png"
    )
    
    return model, metrics

def train_transformer_model(data_loaders, device):
    """
    Train Transformer model.
    
    Args:
        data_loaders: Dictionary of data loaders and vocabularies
        device: Device to run training on
        
    Returns:
        Trained model and training metrics
    """
    print(f"\n{'='*20} Training Transformer Model {'='*20}")
    
    # Create model
    model = TransformerSeq2Seq(
        encoder_vocab_size=len(data_loaders["source_vocab"]),
        decoder_vocab_size=len(data_loaders["target_vocab"]),
        encoder_embedding_dim=config.EMBEDDING_DIM,
        decoder_embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.TRANSFORMER_DIM,
        ff_dim=config.TRANSFORMER_FF_DIM,
        num_layers=config.TRANSFORMER_LAYERS,
        num_heads=config.NUM_HEADS,
        dropout=config.TRANSFORMER_DROPOUT,
        device=device
    ).to(device)
    
    # Set decoder vocabulary
    model.decoder.vocab = data_loaders["target_vocab"]
    
    # Skip model summary for now to avoid errors
    print(f"Model created: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train_loader"],
        val_loader=data_loaders["val_loader"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        clip_grad=config.CLIP_GRAD,
        checkpoint_dir=os.path.join(config.CHECKPOINT_DIR, f"{data_loaders['direction']}_transformer"),
        teacher_forcing_ratio=1.0,  # Always use teacher forcing for Transformer
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Train model
    metrics = trainer.train(config.EPOCHS)
    
    # Plot metrics
    model_name = 'Transformer'
    save_path = f"results/{data_loaders['direction']}_{model_name.lower()}_metrics.png"
    trainer.plot_metrics(save_path)
    
    return model, metrics

def train_llm_model(data, device, direction='en2pur'):
    """
    Fine-tune LLM model.
    
    Args:
        data: Pandas DataFrame with source and target texts
        device: Device to run training on
        direction: Translation direction, either 'en2pur' or 'pur2en'
        
    Returns:
        Trained model and training metrics
    """
    print(f"\n{'='*20} Fine-tuning LLM Model {'='*20}")
    
    # Choose a smaller model for demonstration purposes
    model_name = "t5-small"
    
    # Set source and target columns based on direction
    if direction == 'en2pur':
        source_col = "english"
        target_col = "purepecha"
    else:  # pur2en
        source_col = "purepecha"
        target_col = "english"
    
    # Fine-tune model
    model = fine_tune_llm(
        dataframe=data,
        source_col=source_col,
        target_col=target_col,
        model_name=model_name,
        output_dir=os.path.join(config.CHECKPOINT_DIR, f"{direction}_llm")
    )
    
    return model, None  # No metrics in the same format as other models

def main(args):
    """
    Main function to train and evaluate models.
    """
    # Use global LLM_AVAILABLE
    global LLM_AVAILABLE
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Load and preprocess data
    data_loaders = load_data(
        file_path=config.DATA_PATH,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        batch_size=config.BATCH_SIZE,
        random_seed=config.RANDOM_SEED,
        direction=args.direction
    )
    
    # Load the original data as DataFrame for LLM fine-tuning - using the same format handling as data_utils
    df = None
    if LLM_AVAILABLE:
        try:
            df = pd.read_csv(config.DATA_PATH, sep='\t', header=None)
            
            # Handle different file formats
            if len(df.columns) == 3:
                df.columns = ['id', 'english', 'purepecha']
            elif len(df.columns) == 2:
                df.columns = ['english', 'purepecha']
                df['id'] = range(1, len(df) + 1)
            else:
                if len(df.columns) >= 3:
                    df = df.iloc[:, :3]
                    df.columns = ['id', 'english', 'purepecha']
                else:
                    print("Warning: LLM model will be disabled due to incompatible data format")
                    LLM_AVAILABLE = False
            
            if LLM_AVAILABLE:
                # Ensure all text columns are strings
                for col in ['english', 'purepecha']:
                    df[col] = df[col].astype(str)
                    
                # Drop NaN values if present
                if df['english'].isna().any() or df['purepecha'].isna().any():
                    df = df.dropna(subset=['english', 'purepecha'])
        except Exception as e:
            print(f"Error loading data for LLM: {e}")
            LLM_AVAILABLE = False
    
    # Create models
    models = {}
    
    # Train models based on arguments
    if args.lstm or args.all:
        models["LSTM RNN"], _ = train_rnn_model(data_loaders, device, use_gru=False)
    
    if args.gru or args.all:
        models["GRU RNN"], _ = train_rnn_model(data_loaders, device, use_gru=True)
    
    if args.lstm_attention or args.all:
        models["LSTM with Attention"], _ = train_attention_model(data_loaders, device, use_gru=False)
    
    if args.gru_attention or args.all:
        models["GRU with Attention"], _ = train_attention_model(data_loaders, device, use_gru=True)
    
    if args.transformer or args.all:
        models["Transformer"], _ = train_transformer_model(data_loaders, device)
    
    if (args.llm or args.all) and LLM_AVAILABLE:
        llm_model, _ = train_llm_model(df, device, direction=args.direction)
        # Note: LLM model is not included in comparison due to different API
    elif args.llm:
        print("Cannot train LLM model: dependencies not available")
    elif args.all:
        print("Skipping LLM model: dependencies not available")
    
    # Compare models if multiple models were trained
    if len(models) > 1:
        print(f"\n{'='*20} Comparing Models {'='*20}")
        
        # Compare performance
        results = compare_models(
            models=models,
            test_loader=data_loaders["test_loader"],
            target_vocab=data_loaders["target_vocab"],
            device=device
        )
        
        # Plot comparison
        plot_comparison(results, save_path=f"results/{args.direction}_model_comparison.png")
        
        # Save results to JSON
        with open(f"results/{args.direction}_model_comparison.json", 'w') as f:
            # Convert values to serializable format
            serializable_results = {}
            for model_name, metrics in results.items():
                serializable_results[model_name] = {
                    "bleu": metrics["bleu"],
                    "inference_time": {
                        "mean": metrics["inference_time"]["mean"],
                        "median": metrics["inference_time"]["median"],
                        "min": metrics["inference_time"]["min"],
                        "max": metrics["inference_time"]["max"],
                        "total": metrics["inference_time"]["total"],
                        "num_samples": metrics["inference_time"]["num_samples"]
                    },
                    "model_size": metrics["model_size"],
                    "trainable_params": metrics["trainable_params"]
                }
            
            json.dump(serializable_results, f, indent=4)
        
        print(f"Results saved to results/{args.direction}_model_comparison.json and results/{args.direction}_model_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Purépecha-English translation using neural network models')
    
    # Model selection arguments
    parser.add_argument('--lstm', action='store_true', help='Train LSTM RNN model')
    parser.add_argument('--gru', action='store_true', help='Train GRU RNN model')
    parser.add_argument('--lstm_attention', action='store_true', help='Train LSTM with attention model')
    parser.add_argument('--gru_attention', action='store_true', help='Train GRU with attention model')
    parser.add_argument('--transformer', action='store_true', help='Train Transformer model')
    parser.add_argument('--llm', action='store_true', help='Fine-tune LLM model')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--direction', type=str, choices=['en2pur', 'pur2en'], default='en2pur', 
                       help='Translation direction: en2pur (English to Purépecha) or pur2en (Purépecha to English)')
    
    args = parser.parse_args()
    
    # Default to training all models if none specified
    if not (args.lstm or args.gru or args.lstm_attention or args.gru_attention or 
            args.transformer or args.llm or args.all):
        args.all = True
        
    main(args)