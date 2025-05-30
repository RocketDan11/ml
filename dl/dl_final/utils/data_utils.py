"""
Data utilities for Purépecha-English translation models.
"""
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import random

class Vocabulary:
    """
    Vocabulary class for mapping tokens to indices and vice versa.
    """
    def __init__(self, lang, freq_threshold=1):
        self.lang = lang
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.vocab_size = 4  # Starting with special tokens
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentences):
        """
        Build vocabulary from a list of sentences.
        
        Args:
            sentences: List of sentences to build vocabulary from
        """
        # Count token frequencies
        frequencies = Counter()
        for sentence in sentences:
            for token in self.tokenize(sentence):
                frequencies[token] += 1
                
        # Add tokens that meet frequency threshold to vocabulary
        for token, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[token] = self.vocab_size
                self.itos[self.vocab_size] = token
                self.vocab_size += 1
    
    def tokenize(self, text):
        """
        Tokenize text into list of tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Handle non-string input
        if not isinstance(text, str):
            print(f"Warning: non-string input to tokenize: {text} of type {type(text)}")
            text = str(text)
            
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def numericalize(self, text):
        """
        Convert text to list of indices.
        
        Args:
            text: Text to convert
            
        Returns:
            List of token indices
        """
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class TranslationDataset(Dataset):
    """
    Dataset for sequence-to-sequence translation.
    """
    def __init__(
        self,
        dataframe,
        source_vocab,
        target_vocab,
        source_column,
        target_column,
        max_length=50
    ):
        self.dataframe = dataframe
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_column = source_column
        self.target_column = target_column
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        source_text = str(self.dataframe.iloc[index][self.source_column])
        target_text = str(self.dataframe.iloc[index][self.target_column])
        
        # Convert to numerical form with SOS and EOS tokens
        source_numericalized = self.source_vocab.numericalize(source_text)
        target_numericalized = self.target_vocab.numericalize(target_text)
        
        # Truncate sequences if they exceed max length
        if len(source_numericalized) > self.max_length - 2:  # -2 for SOS and EOS
            source_numericalized = source_numericalized[:self.max_length-2]
            
        if len(target_numericalized) > self.max_length - 2:
            target_numericalized = target_numericalized[:self.max_length-2]
        
        # Add SOS and EOS tokens
        source_numericalized = [self.source_vocab.stoi["<SOS>"]] + source_numericalized + [self.source_vocab.stoi["<EOS>"]]
        target_numericalized = [self.target_vocab.stoi["<SOS>"]] + target_numericalized + [self.target_vocab.stoi["<EOS>"]]
        
        return {
            "source": torch.tensor(source_numericalized),
            "target": torch.tensor(target_numericalized),
            "source_text": source_text,
            "target_text": target_text
        }

def collate_fn(batch):
    """
    Collate function for padding sequences in a batch.
    
    Args:
        batch: List of dictionaries with source and target sequences
        
    Returns:
        Dictionary with padded sequences and lengths
    """
    source_batch = [item["source"] for item in batch]
    target_batch = [item["target"] for item in batch]
    source_text = [item["source_text"] for item in batch]
    target_text = [item["target_text"] for item in batch]
    
    # Get sequence lengths
    source_lengths = torch.tensor([len(seq) for seq in source_batch])
    target_lengths = torch.tensor([len(seq) for seq in target_batch])
    
    # Pad sequences
    source_padded = pad_sequence(source_batch, padding_value=0, batch_first=True)
    target_padded = pad_sequence(target_batch, padding_value=0, batch_first=True)
    
    return {
        "source": source_padded,
        "source_lengths": source_lengths,
        "target": target_padded,
        "target_lengths": target_lengths,
        "source_text": source_text,
        "target_text": target_text
    }

def load_data(file_path, train_split=1.0, val_split=1.0, test_split=1.0, batch_size=64, random_seed=42, direction='en2pur'):
    """
    Load and preprocess translation data.
    
    Args:
        file_path: Path to TSV file containing translation data
        train_split: Proportion of data to use for training (1.0 to use full dataset)
        val_split: Proportion of data to use for validation (1.0 to use full dataset)
        test_split: Proportion of data to use for testing (1.0 to use full dataset)
        batch_size: Batch size for DataLoader
        random_seed: Random seed for reproducibility
        direction: Translation direction, either 'en2pur' (English to Purépecha) or 'pur2en' (Purépecha to English)
        
    Returns:
        Dictionary with DataLoaders and vocabularies
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    # Examine the data to determine the correct columns
    print(f"Data columns: {df.columns}")
    print(f"First few rows of data:\n{df.head()}")
    
    # Handle different file formats
    if len(df.columns) == 3:
        df.columns = ['id', 'english', 'purepecha']
        print("Using 3-column format: id, english, purepecha")
    elif len(df.columns) == 2:
        df.columns = ['english', 'purepecha']
        print("Using 2-column format: english, purepecha")
        # Add an ID column
        df['id'] = range(1, len(df) + 1)
    else:
        print(f"Warning: Unexpected number of columns: {len(df.columns)}")
        # Try to make the best guess
        if len(df.columns) >= 3:
            df = df.iloc[:, :3]
            df.columns = ['id', 'english', 'purepecha']
            print("Assuming first 3 columns are: id, english, purepecha")
        else:
            raise ValueError(f"Cannot determine dataset format with {len(df.columns)} columns")
    
    # Ensure all text columns are strings
    for col in ['english', 'purepecha']:
        df[col] = df[col].astype(str)
        
    # Check for NaN or missing values
    if df['english'].isna().any() or df['purepecha'].isna().any():
        print("Warning: Found NaN values in the dataset")
        df = df.dropna(subset=['english', 'purepecha'])
        print(f"After dropping NaN rows, {len(df)} samples remain")
    
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Use the entire dataset for training, validation, and testing
    train_df = df.copy()
    val_df = df.copy()
    test_df = df.copy()
    
    print(f"Using all {len(df)} samples for training, validation, and testing")
    
    # Initialize and build vocabularies
    purepecha_vocab = Vocabulary("purepecha")
    english_vocab = Vocabulary("english")
    
    print("Building vocabularies...")
    purepecha_vocab.build_vocabulary(train_df["purepecha"].tolist())
    english_vocab.build_vocabulary(train_df["english"].tolist())
    
    print(f"Purépecha vocabulary size: {len(purepecha_vocab)}")
    print(f"English vocabulary size: {len(english_vocab)}")
    
    # Set source and target based on translation direction
    if direction == 'en2pur':
        print("Translation direction: English → Purépecha")
        source_vocab = english_vocab
        target_vocab = purepecha_vocab
        source_col = "english"
        target_col = "purepecha"
    else:  # pur2en
        print("Translation direction: Purépecha → English")
        source_vocab = purepecha_vocab
        target_vocab = english_vocab
        source_col = "purepecha"
        target_col = "english"
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TranslationDataset(
        train_df, source_vocab, target_vocab, source_col, target_col
    )
    val_dataset = TranslationDataset(
        val_df, source_vocab, target_vocab, source_col, target_col
    )
    test_dataset = TranslationDataset(
        test_df, source_vocab, target_vocab, source_col, target_col
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "purepecha_vocab": purepecha_vocab,
        "english_vocab": english_vocab,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "direction": direction
    }