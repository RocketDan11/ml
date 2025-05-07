"""
Global configuration parameters for Purépecha-English translation models.
"""
import torch

# Data parameters
DATA_PATH = "data/purepecha_data.tsv"
MAX_LENGTH = 50
BATCH_SIZE = 64
TRAIN_SPLIT = 1.0
VAL_SPLIT = 1.0
TEST_SPLIT = 1.0
RANDOM_SEED = 42

# Model parameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 3
DROPOUT = 0.3
ATTENTION_DIM = 256
NUM_HEADS = 2
TRANSFORMER_DIM = 512
TRANSFORMER_LAYERS = 4
TRANSFORMER_FF_DIM = 2048
TRANSFORMER_DROPOUT = 0.1

# Training parameters
EPOCHS = 500
LEARNING_RATE = 0.0001
CLIP_GRAD = 1.0
TEACHER_FORCING_RATIO = 0.5
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metrics and evaluation
BLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1, 2, 3, 4-grams