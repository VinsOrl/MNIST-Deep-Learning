# MNIST Deep Learning: CPU vs GPU Comparison

A complete deep learning project using PyTorch to train neural networks on the MNIST handwritten digit dataset. Implements two network architectures and compares performance between CPU and GPU computation.

## Features

- **Two Neural Network Models:**
  - Basic Fully Connected Neural Network (FCNN)
  - FCNN with Dropout regularization

- **Training & Evaluation:**
  - Automated training with metrics tracking
  - Loss and accuracy visualization
  - Sample prediction visualization

- **Performance Benchmarking:**
  - CPU vs GPU training time comparison
  - Inference speed comparison
  - Memory usage tracking

## Requirements

```
Python >= 3.7
torch >= 1.9.0
torchvision >= 0.10.0
numpy
matplotlib
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main training script:
```bash
python mnist_deep_learning.py
```

## Output Files

The script generates the following files:
- `training_curves_simple.png` - Loss & Accuracy for SimpleFCNN
- `training_curves_dropout.png` - Loss & Accuracy for FCNNWithDropout
- `sample_predictions_simple.png` - Sample predictions
- `sample_predictions_dropout.png` - Sample predictions with Dropout
- `cpu_vs_gpu_comparison.png` - Performance comparison chart
- `simple_fcnn.pth` - Saved SimpleFCNN model
- `fcnn_dropout.pth` - Saved FCNNWithDropout model
- `training_history.json` - All training metrics
- `benchmark_results.txt` - CPU vs GPU benchmark results

## Model Architecture

### SimpleFCNN
- Input Layer: 784 neurons (28×28 flattened)
- Hidden Layer 1: 256 neurons + ReLU
- Hidden Layer 2: 128 neurons + ReLU
- Output Layer: 10 neurons (digits 0-9)

### FCNNWithDropout
Same as SimpleFCNN but with `nn.Dropout(p=0.5)` after each hidden layer.

## Expected Results

- **SimpleFCNN Final Accuracy:** 95-97% on test set
- **GPU Speedup:** 5-50x faster depending on GPU model

## License

This project is licensed under the MIT License.
