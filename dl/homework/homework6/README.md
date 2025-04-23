# Swin Transformer Comparison on CIFAR-100

This project compares the performance of different Swin Transformer models on the CIFAR-100 dataset:
1. Pretrained Swin-Tiny (with frozen backbone)
2. Pretrained Swin-Small (with frozen backbone)
3. Swin-Tiny trained from scratch

## Requirements

```
torch
torchvision
transformers
tqdm
pandas
```

Install dependencies:
```
pip install torch torchvision transformers tqdm pandas
```

## Running the Experiment

Run the comparison script:
```
python swin-comparison.py
```

This will:
1. Load the CIFAR-100 dataset
2. Train each model configuration for 3 epochs
3. Measure training time per epoch and test accuracy
4. Save results to a CSV file and display a summary

## Expected Results

The experiment evaluates:
- **Training efficiency**: Time per epoch for each model
- **Transfer learning effectiveness**: How pretrained models compare to training from scratch
- **Model size impact**: Performance difference between Swin-Tiny and Swin-Small

Results will be saved to `swin_comparison_results.csv` and displayed in the terminal.

## Report Analysis

The generated results provide data for analyzing:
- Benefits and drawbacks of fine-tuning vs. training from scratch
- Performance differences between Swin-Tiny and Swin-Small models
- Why pretrained models might outperform or underperform the scratch model

## Notes

- Training from scratch requires significantly more compute resources than fine-tuning
- The experiment uses a small number of epochs (3) to demonstrate the concept
- For production use cases, more epochs and potentially different hyperparameters would be recommended 