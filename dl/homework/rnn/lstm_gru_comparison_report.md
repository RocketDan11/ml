# LSTM vs GRU Comparison Report

This report compares the performance of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models on a character-level language modeling task using the Tiny Shakespeare dataset.

## Dataset
- **Source**: Tiny Shakespeare dataset
- **Vocabulary Size**: 65 unique characters
- **Sequence Length**: 20 characters
- **Training Examples**: ~892,299 sequences (80% of total)
- **Testing Examples**: ~223,075 sequences (20% of total)

## Model Architectures
Both models share the same architecture except for the recurrent layer:
- **Input Size**: 65 (one-hot encoded characters)
- **Hidden Size**: 128
- **Number of Layers**: 2
- **Output Size**: 65 (vocabulary size)

## Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 128
- **Epochs**: 30
- **Loss Function**: Cross Entropy
- **Hardware**: NVIDIA GeForce RTX 3080 Ti GPU

## Results

### Performance Metrics

| Metric | LSTM | GRU | Winner |
|--------|------|-----|--------|
| Final Training Loss | 1.1892 | 1.2705 | LSTM |
| Final Test Loss | 1.3823 | 1.3753 | GRU |
| Final Test Accuracy | 58.16% | 58.08% | LSTM |
| Training Time | 477.15 seconds | 455.48 seconds | GRU |
| Model Size | 0.92 MB | 0.70 MB | GRU |

### Text Generation Examples

**Seed Text**: "The quick brown fox jumps over the lazy dog"

**LSTM Generated Text**:
```
The quick brown fox jumps over the lazy dog.

FRIAR LAURENCE:
O let crown'd at this sun;
And yet not speak! I'll be you at leisure;
If ever I d
```

**GRU Generated Text**:
```
The quick brown fox jumps over the lazy dog,
Some strong compel's lie:
Yea, masterphant makes his conquered will your enmines and his banish'd
```

## Analysis

1. **Accuracy and Loss**: 
   - LSTM achieved slightly better accuracy on the test set (58.16% vs 58.08%)
   - GRU had slightly lower test loss (1.3753 vs 1.3823)
   - LSTM had lower training loss, suggesting it fits the training data better

2. **Efficiency**:
   - GRU was approximately 4.5% faster to train (455.48s vs 477.15s)
   - GRU model is about 24% smaller (0.70 MB vs 0.92 MB)

3. **Text Generation**:
   - Both models generated coherent Shakespeare-like text
   - The quality of generated text is subjective, but both models captured some of the Shakespearean style

## Conclusion

The comparison reveals trade-offs between the two architectures:

- **LSTM** showed marginally better predictive performance with higher accuracy and lower training loss, suggesting it might be capturing more complex patterns in the text.

- **GRU** demonstrated better efficiency with faster training time and smaller model size, while maintaining competitive performance metrics.

For this specific task of character-level language modeling on the Tiny Shakespeare dataset, the differences in performance are minimal. The choice between LSTM and GRU would depend on specific requirements:

- If maximum accuracy is the priority, LSTM might be preferred.
- If efficiency and deployment size are important considerations, GRU would be the better choice.

The results align with the general understanding that GRUs are more efficient while LSTMs can sometimes capture more complex dependencies, though the difference is often task-dependent and can be marginal as seen in this experiment.
