# MNIST Deep Learning: CPU vs GPU Comparison

## Project Overview
Build a complete deep learning project using PyTorch to train neural networks on the MNIST handwritten digit dataset. Implement two network architectures and compare performance between CPU and GPU computation.

---

## Objectives

1. **Build two neural network models**:
   - Model 1: Basic Fully Connected Neural Network (FCNN)
   - Model 2: FCNN with Dropout regularization

2. **Train and evaluate** both models on MNIST dataset with metrics tracking

3. **Perform CPU vs GPU comparison**:
   - Measure training time on CPU
   - Measure training time on GPU
   - Compare inference speed
   - Document performance improvements

4. **Create visualizations**:
   - Training/testing loss curves
   - Training/testing accuracy curves
   - Sample predictions from both models
   - CPU vs GPU speed comparison chart

---

## Dataset: MNIST

- **Size**: 28×28 pixel grayscale images
- **Classes**: 10 (digits 0-9)
- **Training samples**: 60,000
- **Testing samples**: 10,000
- **Source**: PyTorch's `torchvision.datasets.MNIST`

---

## Architecture Specifications

### Model 1: Basic FCNN
```
Input Layer: 784 neurons (28×28 flattened)
    ↓
Hidden Layer 1: 256 neurons + ReLU activation
    ↓
Hidden Layer 2: 128 neurons + ReLU activation
    ↓
Output Layer: 10 neurons (digit classes 0-9)
```

**Components**:
- `nn.Linear(784, 256)` with `nn.ReLU()`
- `nn.Linear(256, 128)` with `nn.ReLU()`
- `nn.Linear(128, 10)` (no activation, handled by loss function)

### Model 2: FCNN with Dropout
**Same as Model 1, but add**:
- `nn.Dropout(p=0.5)` after each hidden layer
- Purpose: Reduce overfitting on test set

---

## Training Configuration

### Hyperparameters
- **Learning Rate (lr)**: 0.001
- **Batch Size**: 64
- **Number of Epochs**: 10-15 (or until convergence)
- **Optimizer**: Adam (`torch.optim.Adam`)
- **Loss Function**: `nn.CrossEntropyLoss()`

### Data Handling
- **Train/Test Split**: Use PyTorch's built-in MNIST dataset
  - `train=True` for training set
  - `train=False` for test set
- **DataLoader**: Use `DataLoader` with `batch_size=64` and `shuffle=True` for training

### Device Management
- **Auto-detect GPU**: 
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- **Move model and data to device**:
  ```python
  model = model.to(device)
  # In training loop: data, labels = data.to(device), labels.to(device)
  ```

---

## Implementation Requirements

### 1. Data Loading & Preprocessing
- Load MNIST using `torchvision.datasets.MNIST`
- Apply transformations:
  - Convert to tensor: `transforms.ToTensor()`
  - Optional: Normalize with mean=0.5, std=0.5
- Create DataLoaders for train and test sets
- Display sample images and labels for verification

### 2. Model Definition (Two Classes)

**Class 1: SimpleFCNN**
```
- Constructor: define layers (Linear + ReLU)
- forward(): sequential forward pass
- Model should be callable: output = model(input_tensor)
```

**Class 2: FCNNWithDropout**
```
- Same as SimpleFCNN but add nn.Dropout(0.5) after each hidden layer
- Dropout should only be active during training (model.train() vs model.eval())
```

### 3. Training Function
```
def train_epoch(model, train_loader, criterion, optimizer, device):
    - Set model to training mode: model.train()
    - Iterate through batches
    - For each batch:
      1. Move data to device
      2. Forward pass: predictions = model(images)
      3. Calculate loss
      4. Zero gradients: optimizer.zero_grad()
      5. Backward pass: loss.backward()
      6. Update weights: optimizer.step()
      7. Track loss and accuracy
    - Return average loss and accuracy for epoch
```

### 4. Evaluation Function
```
def evaluate(model, test_loader, criterion, device):
    - Set model to evaluation mode: model.eval()
    - No gradient calculation: with torch.no_grad():
    - Iterate through batches
    - Calculate predictions, loss, and accuracy
    - Return average loss and accuracy
    - NOTE: Keep Dropout active (model.eval() disables it automatically)
```

### 5. Main Training Loop
```
def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs, device, model_name):
    - Track: train_losses, train_accuracies, test_losses, test_accuracies
    - For each epoch:
      1. Call train_epoch()
      2. Call evaluate()
      3. Print: "Epoch [X/Y], Train Loss: X, Train Acc: X%, Test Loss: X, Test Acc: X%"
      4. Store metrics
    - Return: (model, metrics_dict)
```

### 6. CPU vs GPU Comparison Function
```
def benchmark_cpu_vs_gpu(model, test_loader, criterion, device_list):
    - For each device in [CPU, GPU]:
      1. Move model to device
      2. Measure inference time on test_loader
      3. Calculate average time per batch
      4. Store results
    - Return: comparison_dict with times and speedup ratio
    
    Additional measurements:
    - Training time: Record time.time() before/after training
    - Inference time: Measure time for 100 forward passes
    - Memory usage: torch.cuda.memory_allocated() (if GPU available)
```

### 7. Visualization Functions

**Plot 1: Training & Testing Loss**
```
- X-axis: Epoch number
- Y-axis: Loss value
- Two lines: Train loss (blue), Test loss (orange)
- Title: "Loss Over Epochs"
- Include legend and grid
```

**Plot 2: Training & Testing Accuracy**
```
- X-axis: Epoch number
- Y-axis: Accuracy (%)
- Two lines: Train accuracy (blue), Test accuracy (orange)
- Title: "Accuracy Over Epochs"
- Include legend and grid
```

**Plot 3: Sample Predictions**
```
- Show 10 random test samples in a grid (2 rows × 5 columns)
- For each sample: display image, true label, predicted label
- Color code: green if correct, red if incorrect
```

**Plot 4: CPU vs GPU Performance Comparison**
```
- Bar chart comparing:
  - Training time (CPU vs GPU)
  - Inference time per batch (CPU vs GPU)
  - Speedup ratio (how many times faster GPU is)
- Include values on bars for clarity
```

### 8. Model Saving & Loading
```
- Save best model: torch.save(model.state_dict(), 'best_model.pth')
- Load model: model.load_state_dict(torch.load('best_model.pth'))
- Save training history as JSON or CSV for reference
```

---

## Execution Flow

```
1. Setup & Configuration
   - Set random seeds for reproducibility
   - Configure device (CPU/GPU detection)
   - Define hyperparameters

2. Data Preparation
   - Load MNIST dataset
   - Create DataLoaders
   - Display sample data

3. Model Training (CPU)
   - Initialize SimpleFCNN
   - Train on CPU for N epochs
   - Record all metrics
   - Time the training process

4. Model Evaluation (CPU)
   - Evaluate SimpleFCNN on test set
   - Visualize predictions

5. Model Training (GPU) - if available
   - Initialize SimpleFCNN on GPU
   - Train on GPU for N epochs
   - Record all metrics
   - Time the training process

6. Model with Dropout (GPU preferred, fallback to CPU)
   - Initialize FCNNWithDropout
   - Train for N epochs
   - Record all metrics
   - Compare with Model 1

7. Visualization & Comparison
   - Plot loss curves (both models)
   - Plot accuracy curves (both models)
   - Plot sample predictions
   - Generate CPU vs GPU comparison chart
   - Create summary report

8. Final Output
   - Print summary statistics
   - Save all visualizations as PNG files
   - Save models and training history
   - Print CPU vs GPU performance metrics
```

---

## Key Technical Considerations

### Device Handling
- Always check GPU availability: `torch.cuda.is_available()`
- Tensors and models must be on same device
- Use `device` variable consistently throughout code

### Dropout Behavior
- Automatically disabled in `model.eval()` mode
- Enabled in `model.train()` mode
- Should NOT be used during evaluation/testing

### Loss Calculation
- `nn.CrossEntropyLoss()` expects:
  - Input: raw logits (NOT softmax)
  - Target: class indices (0-9)
- No manual softmax needed

### Gradient Management
- Use `torch.no_grad()` context during evaluation (faster, less memory)
- Always call `optimizer.zero_grad()` before `backward()`

### Reproducibility
- Set random seeds:
  ```python
  torch.manual_seed(42)
  np.random.seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(42)
  ```

---

## Output Requirements

### Console Output
```
Device: GPU (CUDA) / CPU
Number of parameters in SimpleFCNN: X
Number of parameters in FCNNWithDropout: X

Epoch [1/15], Train Loss: 0.XXX, Train Acc: XX.XX%, Test Loss: 0.XXX, Test Acc: XX.XX%
...
Epoch [15/15], Train Loss: 0.XXX, Train Acc: XX.XX%, Test Loss: 0.XXX, Test Acc: XX.XX%

=== CPU vs GPU Benchmark ===
Model: SimpleFCNN
CPU Training Time: X.XX seconds
GPU Training Time: X.XX seconds
Speedup: X.Xx faster on GPU

CPU Inference (per batch): X.XXXs
GPU Inference (per batch): X.XXXs

Model: FCNNWithDropout
[Similar metrics]

=== Final Accuracy ===
SimpleFCNN - Train: XX.XX%, Test: XX.XX%
FCNNWithDropout - Train: XX.XX%, Test: XX.XX%
```

### File Outputs
- `training_curves_simple.png` - Loss & Accuracy for SimpleFCNN
- `training_curves_dropout.png` - Loss & Accuracy for FCNNWithDropout
- `sample_predictions_simple.png` - Sample predictions from SimpleFCNN
- `sample_predictions_dropout.png` - Sample predictions from FCNNWithDropout
- `cpu_vs_gpu_comparison.png` - Performance comparison chart
- `simple_fcnn.pth` - Saved SimpleFCNN model
- `fcnn_dropout.pth` - Saved FCNNWithDropout model
- `training_history.json` - All metrics in JSON format
- `benchmark_results.txt` - CPU vs GPU detailed results

---

## Error Handling

- Check GPU availability gracefully
- Handle missing MNIST dataset (auto-download if needed)
- Validate input shapes match expected dimensions
- Print informative error messages
- Handle keyboard interrupt (Ctrl+C) gracefully

---

## Optional Enhancements (if time permits)

1. **Confusion Matrix**: Show which digits are most confused
2. **Learning Rate Scheduling**: Reduce LR over time
3. **Early Stopping**: Stop training if validation loss plateaus
4. **Batch Normalization**: Add `nn.BatchNorm1d()` layers
5. **More Epochs**: Train for more epochs to see fuller convergence
6. **Hyperparameter Tuning**: Test different learning rates/batch sizes
7. **Memory Profiling**: Track GPU memory usage over time

---

## Testing Checklist

- [ ] Code runs without errors on CPU
- [ ] Code runs without errors on GPU (if available)
- [ ] MNIST data loads correctly
- [ ] Both models train successfully
- [ ] Loss decreases over epochs
- [ ] Accuracy increases over epochs
- [ ] Test accuracy is reasonable (>95%)
- [ ] Dropout model shows less overfitting
- [ ] All visualizations are generated
- [ ] CPU vs GPU comparison is meaningful
- [ ] Models can be saved and loaded
- [ ] Code handles device transfers correctly

---

## Expected Results

- **SimpleFCNN Final Accuracy**: 95-97% on test set
- **FCNNWithDropout Final Accuracy**: 95-97% on test set (often slightly lower on train, better on test)
- **GPU Speedup**: 5-50x faster depending on GPU model (RTX 3090 >> Tesla V100 >> GTX 1080)
- **Training Time**: 10-30 seconds per epoch on GPU, 1-3 minutes per epoch on CPU

---

## Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
```

Install with: `pip install torch torchvision numpy matplotlib`

---

## Code Style Requirements

- Clear variable names
- Comments for non-obvious logic
- Type hints for function arguments (optional but appreciated)
- Consistent indentation (4 spaces)
- No hardcoded values (use constants at top)
- Functions should have docstrings

---

## Final Deliverables

1. ✅ Complete working Python script
2. ✅ 4 Training curve visualizations (both models)
3. ✅ Sample prediction visualizations (both models)
4. ✅ CPU vs GPU performance comparison chart
5. ✅ Saved model weights
6. ✅ Training history data
7. ✅ Console output showing all metrics
8. ✅ Benchmark report with timing details
