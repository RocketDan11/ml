"""
Fine-tuned LLM model for sequence-to-sequence translation.
Uses the Hugging Face Transformers library to fine-tune pre-trained language models.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset as HFDataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

class LLMTranslationModel:
    """
    Fine-tuned LLM model for sequence-to-sequence translation.
    """
    def __init__(
        self,
        model_name="facebook/mbart-large-50",  # or "t5-small", "google/mt5-small", etc.
        source_lang="purepecha",
        target_lang="english",
        output_dir="results/llm",
        device=None
    ):
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_dir = output_dir
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = 128
        self.max_target_length = 128
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # For some multilingual models, we need to set source and target language tokens
        if "mbart" in model_name:
            # Map to mBART language codes
            self.tokenizer.src_lang = "en_XX"  # Use English as proxy for Purépecha
            self.tokenizer.tgt_lang = "en_XX"
        
    def preprocess_function(self, examples):
        """
        Preprocess examples for training.
        
        Args:
            examples: Dictionary with source and target texts
            
        Returns:
            Dictionary with processed inputs and labels
        """
        # Tokenize inputs
        inputs = [text for text in examples[self.source_lang]]
        targets = [text for text in examples[self.target_lang]]
        
        # Special handling for mBART
        if "mbart" in self.model_name:
            model_inputs = self.tokenizer(
                inputs, max_length=self.max_input_length, truncation=True, padding="max_length"
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets, max_length=self.max_target_length, truncation=True, padding="max_length"
                )
        else:
            # For T5-like models
            prefix = f"translate {self.source_lang} to {self.target_lang}: "
            model_inputs = self.tokenizer(
                [prefix + text for text in inputs],
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length"
            )
            labels = self.tokenizer(
                targets, max_length=self.max_target_length, truncation=True, padding="max_length"
            )
        
        # Replace padding token id with -100 for labels (to ignore in loss calculation)
        labels["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_dataset(self, dataframe, column_mapping=None):
        """
        Prepare dataset for fine-tuning.
        
        Args:
            dataframe: Pandas DataFrame with source and target texts
            column_mapping: Dictionary mapping dataframe columns to source/target languages
            
        Returns:
            Dictionary with train, validation, and test datasets
        """
        # Default column mapping
        if column_mapping is None:
            column_mapping = {
                self.source_lang: "purepecha",
                self.target_lang: "english"
            }
            
        # Rename columns
        df = dataframe.rename(columns=column_mapping)
        
        # Convert to Hugging Face Dataset
        hf_dataset = HFDataset.from_pandas(df)
        
        # Preprocess dataset
        tokenized_dataset = hf_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=hf_dataset.column_names
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_preds):
        """
        Compute metrics for evaluation.
        
        Args:
            eval_preds: Tuple of predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_preds
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        
        # Replace -100 in labels with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode labels
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        
        # Convert to list of tokens
        pred_tokens = [pred.split() for pred in decoded_preds]
        label_tokens = [label.split() for label in decoded_labels]
        
        # Use nltk.bleu to compute BLEU score with smoothing
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method3
        bleu_score = corpus_bleu([[l] for l in label_tokens], pred_tokens, smoothing_function=smoothie)
        
        return {"bleu": bleu_score}
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=1000,
        logging_steps=100
    ):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_steps: Steps between saving checkpoints
            logging_steps: Steps between logging
            
        Returns:
            Training metrics
        """
        # Define training arguments with minimal parameters to avoid compatibility issues
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_epochs,
            save_total_limit=2,
            logging_steps=logging_steps,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            report_to="none"  # Disable wandb, tensorboard, etc.
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding=True
        )
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if val_dataset is not None else None
        )
        
        # Fine-tune model
        trainer.train()
        
        # Save best model
        trainer.save_model(os.path.join(self.output_dir, "best"))
        
        return trainer.state.log_history
    
    def translate(self, text, max_length=50, num_beams=4):
        """
        Translate a single text.
        
        Args:
            text: Text to translate
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search
            
        Returns:
            Translated text
        """
        self.model.eval()
        
        # Tokenize input
        if "mbart" in self.model_name:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_input_length, truncation=True).to(self.device)
            # Set source language
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        else:
            # For T5-like models
            prefix = f"translate {self.source_lang} to {self.target_lang}: "
            inputs = self.tokenizer(prefix + text, return_tensors="pt", max_length=self.max_input_length, truncation=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
    
    def translate_batch(self, texts, max_length=50, num_beams=4, batch_size=16):
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            max_length: Maximum length of generated translations
            num_beams: Number of beams for beam search
            batch_size: Batch size for inference
            
        Returns:
            List of translated texts
        """
        self.model.eval()
        translations = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            if "mbart" in self.model_name:
                inputs = self.tokenizer(batch_texts, return_tensors="pt", max_length=self.max_input_length, 
                                       truncation=True, padding=True).to(self.device)
            else:
                # For T5-like models
                prefix = f"translate {self.source_lang} to {self.target_lang}: "
                inputs = self.tokenizer([prefix + text for text in batch_texts], 
                                       return_tensors="pt", max_length=self.max_input_length, 
                                       truncation=True, padding=True).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode outputs
            batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations
    
    def evaluate_bleu(self, test_dataset, source_texts_key=None, reference_texts=None):
        """
        Evaluate BLEU score on test dataset.
        
        Args:
            test_dataset: Test dataset (either HF Dataset or list of source texts)
            source_texts_key: Key for source texts in HF Dataset
            reference_texts: List of reference translations
            
        Returns:
            Dictionary with BLEU score and translations
        """
        # Get source texts
        if source_texts_key is not None and isinstance(test_dataset, HFDataset):
            source_texts = test_dataset[source_texts_key]
        else:
            source_texts = test_dataset
        
        # Translate texts
        translations = self.translate_batch(source_texts)
        
        # Compute BLEU score if reference texts are provided
        bleu_score = 0.0
        if reference_texts is not None:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            
            # Tokenize translations and references
            translations_tokens = [t.split() for t in translations]
            reference_tokens = [[r.split()] for r in reference_texts]
            
            # Apply smoothing
            smoothie = SmoothingFunction().method3
            bleu_score = corpus_bleu(reference_tokens, translations_tokens, smoothing_function=smoothie)
            
            print(f"\nBLEU Score for LLM model: {bleu_score:.4f}")
            
            # Print some example translations
            num_examples = min(5, len(translations))
            print("\nExample translations:")
            for i in range(num_examples):
                print(f"Source: {source_texts[i]}")
                print(f"Reference: {reference_texts[i]}")
                print(f"Translation: {translations[i]}")
                print("---")
        
        return {
            "bleu": bleu_score,
            "translations": translations,
            "sources": source_texts[:10],
            "references": reference_texts[:10] if reference_texts is not None else None
        }
    
    def save(self, path=None):
        """
        Save the model and tokenizer.
        
        Args:
            path: Path to save the model and tokenizer
        """
        if path is None:
            path = os.path.join(self.output_dir, "final")
            
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load(self, path=None):
        """
        Load the model and tokenizer.
        
        Args:
            path: Path to load the model and tokenizer from
        """
        if path is None:
            path = os.path.join(self.output_dir, "best")
            
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # For some multilingual models, we need to set source and target language tokens
        if "mbart" in self.model_name:
            self.tokenizer.src_lang = "en_XX"  # Use English as proxy for Purépecha
            self.tokenizer.tgt_lang = "en_XX"

def fine_tune_llm(dataframe, source_col="english", target_col="purepecha", model_name="t5-small", output_dir="results/llm"):
    """
    Fine-tune an LLM for translation.
    
    Args:
        dataframe: Pandas DataFrame with source and target texts
        source_col: Column name for source texts (English)
        target_col: Column name for target texts (Purépecha)
        model_name: HuggingFace model name
        output_dir: Directory to save results
        
    Returns:
        Trained model
    """
    # Create LLM translation model
    model = LLMTranslationModel(
        model_name=model_name,
        source_lang=source_col,
        target_lang=target_col,
        output_dir=output_dir
    )
    
    # Use the entire dataset for training, validation, and testing
    train_df = dataframe.copy()
    val_df = dataframe.copy()
    test_df = dataframe.copy()
    
    print(f"Using all {len(dataframe)} samples for training, validation, and testing")
    
    # Prepare datasets
    column_mapping = {
        source_col: source_col,
        target_col: target_col
    }
    
    train_dataset = model.prepare_dataset(train_df, column_mapping)
    val_dataset = model.prepare_dataset(val_df, column_mapping)
    test_dataset = model.prepare_dataset(test_df, column_mapping)
    
    # Fine-tune model
    training_metrics = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_epochs=10,
        learning_rate=5e-5
    )
    
    # Evaluate model
    evaluation_results = model.evaluate_bleu(
        test_dataset=test_df[source_col].tolist(),
        reference_texts=test_df[target_col].tolist()
    )
    
    # Save evaluation results
    import json
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        # Create serializable results
        serializable_results = {
            "bleu": evaluation_results["bleu"],
            "example_translations": [
                {
                    "source": src,
                    "reference": ref,
                    "translation": trans
                }
                for src, ref, trans in zip(
                    evaluation_results["sources"],
                    evaluation_results["references"],
                    evaluation_results["translations"][:10]
                )
            ]
        }
        json.dump(serializable_results, f, indent=4)
    
    # Save model
    model.save()
    
    return model