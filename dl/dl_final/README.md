# Purépecha-English Neural Machine Translation

This project implements and compares different neural network architectures for translating between Purépecha and English. The following models are implemented and evaluated:

1. LSTM/GRU Recurrent Neural Networks
2. RNNs with Attention Mechanism 
3. Transformer Models
4. Fine-tuned Pre-trained Language Models

## Dataset

The dataset consists of Purépecha-English sentence pairs provided in a TSV file format. The dataset contains basic vocabulary, phrases, and sentences in both languages.

## Project Structure

```
.
├── config.py                # Global configuration parameters
├── data/                    # Data directory
│   └── purepecha_data.tsv   # Dataset file
├── main.py                  # Main script to run training and evaluation
├── models/                  # Model implementations
│   ├── attention_model.py   # RNN with attention model
│   ├── llm_model.py         # Fine-tuned LLM model
│   ├── rnn_model.py         # LSTM/GRU RNN model
│   └── transformer_model.py # Transformer model
├── requirements.txt         # Python dependencies
├── results/                 # Results directory (created during execution)
└── utils/                   # Utility functions
    ├── data_utils.py        # Data preprocessing utilities
    ├── metrics.py           # Evaluation metrics
    └── train_utils.py       # Training utilities
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate all models:

```bash
python main.py --all
```

To train specific models:

```bash
python main.py --lstm --transformer  # Train only LSTM and Transformer models
```

### Translation Direction

You can specify the translation direction using the `--direction` parameter:

```bash
python main.py --all --direction en2pur  # English to Purépecha (default)
python main.py --all --direction pur2en  # Purépecha to English
```

Available options:
- `--lstm`: Train LSTM RNN model
- `--gru`: Train GRU RNN model
- `--lstm_attention`: Train LSTM with attention model
- `--gru_attention`: Train GRU with attention model
- `--transformer`: Train Transformer model
- `--llm`: Fine-tune LLM model
- `--all`: Train all models (default if no options provided)
- `--direction`: Translation direction, either `en2pur` (English to Purépecha, default) or `pur2en` (Purépecha to English)

## Model Evaluation

The models are evaluated on the following metrics:

1. BLEU Score: Measures translation quality
2. Inference Speed: Time taken to translate sentences
3. Model Size: Number of parameters in the model

Results are saved in the `results/` directory:
- Training and validation metrics for each model
- Attention visualizations for attention-based models
- Model comparison results and visualization

Results are organized by translation direction, with filenames prefixed with the direction (e.g., `en2pur_model_comparison.json` or `pur2en_model_comparison.png`).

## Translation Direction and BLEU Scores

It's important to note that BLEU scores may differ significantly between translation directions:
- Translating from English to Purépecha (en2pur) typically produces lower BLEU scores as Purépecha is a morphologically complex language with fewer computational resources.
- Translating from Purépecha to English (pur2en) often results in higher BLEU scores due to English being a resource-rich language with simpler morphology.

Comparing performance in both directions provides a more comprehensive evaluation of model capabilities.

## Customization

Model hyperparameters and training settings can be modified in the `config.py` file, including:

- Embedding dimensions
- Hidden dimensions
- Number of layers
- Learning rate
- Batch size
- Training epochs
- Early stopping patience

## Results

After training, the models are compared based on their performance metrics. The comparison results are saved in `results/{direction}_model_comparison.json` and visualized in `results/{direction}_model_comparison.png`, where `{direction}` is either `en2pur` or `pur2en`.

## References

- Attention Is All You Need (Transformer architecture): https://arxiv.org/abs/1706.03762
- Sequence to Sequence Learning with Neural Networks: https://arxiv.org/abs/1409.3215
- Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473