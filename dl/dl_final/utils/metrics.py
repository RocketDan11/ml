"""
Metrics utilities for translation models.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from tqdm import tqdm

def calculate_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculate BLEU score for a single sentence.
    
    Args:
        reference: Reference sentence (list of tokens)
        hypothesis: Hypothesis sentence (list of tokens)
        weights: Weights for n-grams (default: equal weights for 1-4 grams)
        
    Returns:
        BLEU score
    """
    # Apply smoothing using method3 to handle cases with no higher n-gram matches
    smoothie = SmoothingFunction().method3
    return sentence_bleu([reference], hypothesis, weights=weights, smoothing_function=smoothie)

def evaluate_model_bleu(model, data_loader, target_vocab, device):
    """
    Evaluate the model's BLEU score on a dataset.
    
    Args:
        model: Trained translation model
        data_loader: DataLoader for dataset
        target_vocab: Target vocabulary
        device: Device to run inference on
        
    Returns:
        Dictionary with BLEU score and reference/hypothesis samples
    """
    model.eval()
    total_bleu = 0
    samples = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating BLEU"):
            src = batch["source"].to(device)
            src_lengths = batch["source_lengths"].to(device)
            
            # Process each sample in the batch separately to avoid batch processing issues
            for i in range(src.size(0)):
                # Get individual sample
                single_src = src[i:i+1]
                single_src_length = src_lengths[i:i+1]
                
                # Generate translation
                translation = model.translate(single_src, single_src_length)
                
                # Get reference translation (actual target)
                reference = batch["target_text"][i].split()
                
                # Get hypothesis (predicted translation)
                hypothesis = []
                for idx in translation:
                    token = target_vocab.itos[idx]
                    if token == "<EOS>":
                        break
                    if token not in ["<SOS>", "<PAD>", "<UNK>"]:
                        hypothesis.append(token)
                
                # Calculate BLEU score
                bleu = calculate_bleu(reference, hypothesis)
                total_bleu += bleu
                sample_count += 1
                
                # Save some samples
                if len(samples) < 10:
                    samples.append({
                        "source": batch["source_text"][i],
                        "reference": " ".join(reference),
                        "hypothesis": " ".join(hypothesis),
                        "bleu": bleu
                    })
    
    avg_bleu = total_bleu / sample_count
    
    return {
        "bleu": avg_bleu,
        "samples": samples
    }

def measure_inference_time(model, data_loader, device, num_samples=100):
    """
    Measure inference time for the model.
    
    Args:
        model: Trained translation model
        data_loader: DataLoader for dataset
        device: Device to run inference on
        num_samples: Number of samples to use for measurement
        
    Returns:
        Dictionary with inference time metrics
    """
    model.eval()
    times = []
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch["source"].to(device)
            src_lengths = batch["source_lengths"].to(device)
            
            # Measure time for each sample in the batch
            for i in range(min(src.size(0), num_samples - len(times))):
                single_src = src[i:i+1]
                single_src_length = src_lengths[i:i+1]
                
                start_time = time.time()
                model.translate(single_src, single_src_length)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if len(times) >= num_samples:
                    break
            
            if len(times) >= num_samples:
                break
    
    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "total": sum(times),
        "num_samples": len(times)
    }

def compare_models(models, test_loader, target_vocab, device):
    """
    Compare multiple models on performance metrics.
    
    Args:
        models: Dictionary of model names to models
        test_loader: DataLoader for test set
        target_vocab: Target vocabulary
        device: Device to run inference on
        
    Returns:
        Dictionary with comparison metrics
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Measure BLEU score
        bleu_result = evaluate_model_bleu(model, test_loader, target_vocab, device)
        
        # Measure inference time
        time_result = measure_inference_time(model, test_loader, device)
        
        # Calculate model size and complexity
        model_size = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results[name] = {
            "bleu": bleu_result["bleu"],
            "inference_time": time_result,
            "model_size": model_size,
            "trainable_params": trainable_params,
            "samples": bleu_result["samples"]
        }
    
    return results

def plot_comparison(results, save_path=None):
    """
    Plot comparison of model performance.
    
    Args:
        results: Dictionary with comparison metrics
        save_path: Path to save the plot
    """
    model_names = list(results.keys())
    bleu_scores = [results[name]["bleu"] for name in model_names]
    inference_times = [results[name]["inference_time"]["mean"] * 1000 for name in model_names]  # ms
    model_sizes = [results[name]["model_size"] / 1e6 for name in model_names]  # millions
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot BLEU scores
    ax[0].bar(model_names, bleu_scores, color='skyblue')
    ax[0].set_title('BLEU Score (higher is better)')
    ax[0].set_ylabel('BLEU Score')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(bleu_scores):
        ax[0].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # Plot inference times
    ax[1].bar(model_names, inference_times, color='salmon')
    ax[1].set_title('Inference Time (lower is better)')
    ax[1].set_ylabel('Time (ms)')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(inference_times):
        ax[1].text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    # Plot model sizes
    ax[2].bar(model_names, model_sizes, color='lightgreen')
    ax[2].set_title('Model Size (lower is better)')
    ax[2].set_ylabel('Parameters (millions)')
    ax[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(model_sizes):
        ax[2].text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_attention(model, sentence, source_vocab, target_vocab, device, save_path=None):
    """
    Visualize attention weights for a translation.
    
    Args:
        model: Trained translation model with attention
        sentence: Source sentence to translate
        source_vocab: Source vocabulary (English)
        target_vocab: Target vocabulary (Purépecha)
        device: Device to run inference on
        save_path: Path to save the visualization
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
        # Generate translation and get attention weights
        translation, attention = model.translate_with_attention(source_tensor, source_length)
    
    # Convert output indices to tokens
    translated_tokens = []
    for idx in translation:
        token = target_vocab.itos[idx]
        if token == "<EOS>":
            break
        if token not in ["<SOS>", "<PAD>", "<UNK>"]:
            translated_tokens.append(token)
    
    # Get source tokens (excluding <SOS> and <EOS>)
    source_tokens = [source_vocab.itos[idx] for idx in indices[1:-1]]
    
    # Plot attention weights
    plt.figure(figsize=(10, 8))
    attention = attention.squeeze(0).cpu().numpy()
    plt.imshow(attention[:len(translated_tokens), :len(source_tokens)], cmap='viridis')
    
    # Set labels
    plt.xticks(range(len(source_tokens)), source_tokens, rotation=90)
    plt.yticks(range(len(translated_tokens)), translated_tokens)
    
    plt.xlabel('Source')
    plt.ylabel('Translation')
    plt.title('Attention Weights')
    plt.colorbar()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return {
        "source": " ".join(source_tokens),
        "translation": " ".join(translated_tokens),
        "attention": attention
    }