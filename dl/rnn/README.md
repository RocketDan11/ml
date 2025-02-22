# Simple RNN Implementation in PyTorch

This project implements a simple Recurrent Neural Network (RNN) using PyTorch's `nn.RNN` module.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The `SimpleRNN` class provides a basic RNN implementation with the following features:
- Configurable input size, hidden size, and number of layers
- Batch-first processing
- Fully connected output layer

### Example Usage

```python
from rnn_model import SimpleRNN
import torch

# Create model instance
model = SimpleRNN(
    input_size=4,    # Size of input features
    hidden_size=10,  # Number of features in hidden state
    num_layers=2,    # Number of recurrent layers
    output_size=1    # Size of output features
)

# Create example input (batch_size, sequence_length, input_size)
x = torch.randn(2, 3, 4)

# Forward pass
output, hidden = model(x)
```

The model expects input tensors of shape `(batch_size, sequence_length, input_size)` and returns:
- output: tensor of shape `(batch_size, sequence_length, output_size)`
- hidden: final hidden state of shape `(num_layers, batch_size, hidden_size)`
