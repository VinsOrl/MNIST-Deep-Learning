"""
MNIST Deep Learning: CPU vs GPU Comparison
A complete deep learning project using PyTorch to train neural networks on MNIST.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
INPUT_SIZE = 784  # 28x28 flattened
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128
NUM_CLASSES = 10

# =============================================================================
# Data Loading & Preprocessing
# =============================================================================
print("\n" + "="*60)
print("Loading MNIST Dataset...")
print("="*60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches - Train: {len(train_loader)}, Test: {len(test_loader)}")


def display_sample_images(dataloader, num_images=10):
    """Display sample images from the dataset."""
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_sample_images.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Sample images saved: mnist_sample_images.png")


display_sample_images(train_loader)

# =============================================================================
# Model 1: Basic FCNN
# =============================================================================
class SimpleFCNN(nn.Module):
    """Basic Fully Connected Neural Network for MNIST."""

    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN_SIZE_1,
                 hidden2=HIDDEN_SIZE_2, num_classes=NUM_CLASSES):
        super(SimpleFCNN, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# =============================================================================
# Model 2: FCNN with Dropout
# =============================================================================
class FCNNWithDropout(nn.Module):
    """FCNN with Dropout regularization to reduce overfitting."""

    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN_SIZE_1,
                 hidden2=HIDDEN_SIZE_2, num_classes=NUM_CLASSES, dropout_p=0.5):
        super(FCNNWithDropout, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Initialize models
model1 = SimpleFCNN().to(DEVICE)
model2 = FCNNWithDropout().to(DEVICE)

print(f"\nModel 1 (SimpleFCNN) parameters: {count_parameters(model1):,}")
print(f"Model 2 (FCNNWithDropout) parameters: {count_parameters(model2):,}")

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)
optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)

# =============================================================================
# Training Function
# =============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        total_correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy


# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy, all_predictions, all_targets


# =============================================================================
# Main Training Loop
# =============================================================================
def train_model(model, train_loader, test_loader, criterion, optimizer,
                num_epochs, device, model_name):
    """Train model for multiple epochs and track metrics."""
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_acc = 0
    best_state_dict = None

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch:2d}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'best_test_acc': best_test_acc
    }

    return model, metrics


# =============================================================================
# Benchmark CPU vs GPU
# =============================================================================
def benchmark_cpu_vs_gpu(model, train_loader, test_loader, criterion, device, num_epochs=5):
    """Benchmark training and inference time on CPU vs GPU."""
    # Train on current device
    device_name = str(device) if isinstance(device, torch.device) else device
    model_train = type(model)().to(device)
    optimizer = optim.Adam(model_train.parameters(), lr=LEARNING_RATE)

    # Measure training time
    start_time = time.time()
    for epoch in range(num_epochs):
        train_epoch(model_train, train_loader, criterion, optimizer, device)
    training_time = time.time() - start_time

    # Measure inference time (average over 100 forward passes)
    inference_times = []
    model_train.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= 100:  # Limit to 100 batches
                break
            data = data.to(device)
            start_infer = time.time()
            _ = model_train(data)
            inference_times.append(time.time() - start_infer)

    avg_inference_time = np.mean(inference_times)

    # GPU memory usage if available
    gpu_memory = None
    if device_name == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    results = {
        'device': device_name,
        'training_time_seconds': training_time,
        'avg_inference_time': avg_inference_time,
        'gpu_memory_mb': gpu_memory
    }

    return results


def compare_cpu_gpu(model_class, train_loader, test_loader, criterion, device):
    """Compare performance between CPU and GPU."""
    print(f"\n{'='*60}")
    print("CPU vs GPU Benchmark")
    print(f"{'='*60}")

    results = {}

    # Benchmark on current device (already set to CPU or GPU)
    device_name = str(device)
    if device_name == 'cuda':
        device_name = 'cuda'  # Ensure consistent key

    results[device_name] = benchmark_cpu_vs_gpu(
        model_class().to(device), train_loader, test_loader, criterion, device, num_epochs=3
    )

    # If GPU available, also benchmark CPU
    if torch.cuda.is_available() and device_name == 'cuda':
        print("\nBenchmarking on CPU...")
        results['cpu'] = benchmark_cpu_vs_gpu(
            model_class().to('cpu'), train_loader, test_loader, criterion, 'cpu', num_epochs=3
        )

        # Print comparison
        gpu_time = results['cuda']['training_time_seconds']
        cpu_time = results['cpu']['training_time_seconds']
        speedup = cpu_time / gpu_time

        print(f"\n{'Device':<15} {'Training Time':<20} {'Inference Time':<20}")
        print("-" * 55)
        print(f"{'CPU':<15} {cpu_time:.2f}s {'':<10} {results['cpu']['avg_inference_time']:.6f}s")
        print(f"{'GPU':<15} {gpu_time:.2f}s {'':<10} {results['cuda']['avg_inference_time']:.6f}s")
        print(f"\nSpeedup: {speedup:.2f}x faster on GPU")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_training_curves(metrics, model_name, save_path):
    """Plot training and testing loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(metrics['train_losses']) + 1)

    # Plot 1: Loss
    axes[0].plot(epochs, metrics['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, metrics['test_losses'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss Over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(epochs, metrics['train_accuracies'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, metrics['test_accuracies'], 'r-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy Over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_predictions(model, test_loader, device, model_name, save_path, num_images=10):
    """Display sample predictions with correct/incorrect coloring."""
    model.eval()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_images].to(device), labels[:num_images]

    with torch.no_grad():
        outputs = model(images[:num_images])
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].cpu().squeeze().numpy()
        ax = axes[i]
        ax.imshow(img, cmap='gray')

        true_label = labels[i].item()
        pred_label = predicted[i].item()

        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'T:{true_label}\nP:{pred_label}', color=color, fontsize=12)
        ax.axis('off')

    plt.suptitle(f'{model_name} - Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cpu_gpu_comparison(results, save_path):
    """Create bar chart comparing CPU and GPU performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Check if we have both CPU and GPU results
    has_gpu = 'cuda' in results or 'GPU' in str(results)
    has_cpu = 'cpu' in results or 'CPU' in str(results)

    if has_cpu and has_gpu:
        devices = ['CPU', 'GPU']
        training_times = [results['cpu']['training_time_seconds'],
                         results['cuda']['training_time_seconds']]
        inference_times = [results['cpu']['avg_inference_time'],
                          results['cuda']['avg_inference_time']]
        speedup = training_times[0] / training_times[1]
    else:
        # Single device results
        device_name = list(results.keys())[0]
        devices = [device_name]
        training_times = [results[device_name]['training_time_seconds']]
        inference_times = [results[device_name]['avg_inference_time']]
        speedup = 1.0

    # Plot 1: Training Time
    colors = ['#FF6B6B', '#4ECDC4'] if len(devices) == 2 else ['#4ECDC4']
    bars1 = axes[0].bar(devices, training_times, color=colors)
    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].set_title('Training Time Comparison')
    for bar, time_val in zip(bars1, training_times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontsize=11)

    # Plot 2: Inference Time
    bars2 = axes[1].bar(devices, inference_times, color=colors)
    axes[1].set_ylabel('Inference Time (seconds)')
    axes[1].set_title('Inference Time Comparison')
    for bar, time_val in zip(bars2, inference_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                    f'{time_val:.6f}', ha='center', va='bottom', fontsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return speedup


# =============================================================================
# Save Utilities
# =============================================================================
def save_model(model, save_path):
    """Save model state dictionary."""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")


def save_training_history(metrics, save_path):
    """Save training metrics as JSON."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training history saved: {save_path}")


def save_benchmark_results(results, save_path):
    """Save benchmark results to text file."""
    with open(save_path, 'w') as f:
        f.write("CPU vs GPU Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        for device_name, device_results in results.items():
            # Handle both direct dict and nested structure
            if 'device' in device_results:
                device_results = device_results  # Keep as is
            f.write(f"Device: {device_name}\n")
            f.write(f"  Training Time: {device_results.get('training_time_seconds', 0):.2f} seconds\n")
            f.write(f"  Inference Time (per batch): {device_results.get('avg_inference_time', 0):.6f} seconds\n")
            if device_results.get('gpu_memory_mb'):
                f.write(f"  GPU Memory: {device_results['gpu_memory_mb']:.2f} MB\n")
            f.write("\n")
    print(f"Benchmark results saved: {save_path}")


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Main function to run the entire training pipeline."""
    print("\n" + "="*60)
    print("MNIST Deep Learning - CPU vs GPU Comparison")
    print("="*60)

    # Storage for all results
    all_metrics = {}
    benchmark_results = {}

    # ========================================
    # Model 1: SimpleFCNN
    # ========================================
    print("\n" + "="*60)
    print("MODEL 1: SimpleFCNN")
    print("="*60)

    model1 = SimpleFCNN().to(DEVICE)
    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)

    # Benchmark CPU vs GPU
    benchmark_results['SimpleFCNN'] = compare_cpu_gpu(
        SimpleFCNN, train_loader, test_loader, criterion, DEVICE
    )

    # Train the model
    model1, metrics1 = train_model(
        model1, train_loader, test_loader, criterion, optimizer1,
        NUM_EPOCHS, DEVICE, "SimpleFCNN"
    )
    all_metrics['SimpleFCNN'] = metrics1

    # Save SimpleFCNN model
    save_model(model1, 'simple_fcnn.pth')

    # Generate visualizations for SimpleFCNN
    plot_training_curves(metrics1, 'SimpleFCNN', 'training_curves_simple.png')
    plot_sample_predictions(model1, test_loader, DEVICE, 'SimpleFCNN',
                          'sample_predictions_simple.png')

    # ========================================
    # Model 2: FCNNWithDropout
    # ========================================
    print("\n" + "="*60)
    print("MODEL 2: FCNNWithDropout")
    print("="*60)

    model2 = FCNNWithDropout().to(DEVICE)
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)

    # Benchmark CPU vs GPU
    benchmark_results['FCNNWithDropout'] = compare_cpu_gpu(
        FCNNWithDropout, train_loader, test_loader, criterion, DEVICE
    )

    # Train the model
    model2, metrics2 = train_model(
        model2, train_loader, test_loader, criterion, optimizer2,
        NUM_EPOCHS, DEVICE, "FCNNWithDropout"
    )
    all_metrics['FCNNWithDropout'] = metrics2

    # Save FCNNWithDropout model
    save_model(model2, 'fcnn_dropout.pth')

    # Generate visualizations for FCNNWithDropout
    plot_training_curves(metrics2, 'FCNNWithDropout', 'training_curves_dropout.png')
    plot_sample_predictions(model2, test_loader, DEVICE, 'FCNNWithDropout',
                          'sample_predictions_dropout.png')

    # ========================================
    # Create CPU vs GPU Comparison Chart
    # ========================================
    print("\n" + "="*60)
    print("Creating CPU vs GPU Comparison")
    print("="*60)

    # Use SimpleFCNN results for comparison chart
    speedup = plot_cpu_gpu_comparison(benchmark_results['SimpleFCNN'],
                                    'cpu_vs_gpu_comparison.png')

    # ========================================
    # Save Training History
    # ========================================
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    combined_metrics = {
        'SimpleFCNN': {
            'train_losses': metrics1['train_losses'],
            'train_accuracies': metrics1['train_accuracies'],
            'test_losses': metrics1['test_losses'],
            'test_accuracies': metrics1['test_accuracies'],
            'best_test_acc': metrics1['best_test_acc'],
            'final_train_loss': metrics1['train_losses'][-1],
            'final_train_acc': metrics1['train_accuracies'][-1],
            'final_test_loss': metrics1['test_losses'][-1],
            'final_test_acc': metrics1['test_accuracies'][-1]
        },
        'FCNNWithDropout': {
            'train_losses': metrics2['train_losses'],
            'train_accuracies': metrics2['train_accuracies'],
            'test_losses': metrics2['test_losses'],
            'test_accuracies': metrics2['test_accuracies'],
            'best_test_acc': metrics2['best_test_acc'],
            'final_train_loss': metrics2['train_losses'][-1],
            'final_train_acc': metrics2['train_accuracies'][-1],
            'final_test_loss': metrics2['test_losses'][-1],
            'final_test_acc': metrics2['test_accuracies'][-1]
        }
    }
    save_training_history(combined_metrics, 'training_history.json')

    # ========================================
    # Save Benchmark Results
    # ========================================
    save_benchmark_results(benchmark_results, 'benchmark_results.txt')

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"\n{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Train Loss':<12} {'Test Loss':<12}")
    print("-" * 73)

    for name, metrics in combined_metrics.items():
        print(f"{name:<25} {metrics['final_train_acc']:.2f}% {metrics['final_test_acc']:.2f}% "
              f"{metrics['final_train_loss']:.4f} {metrics['final_test_loss']:.4f}")

    # GPU check
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(DEVICE)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

    print(f"\nSpeedup (GPU vs CPU): {speedup:.2f}x")

    # Check for overfitting
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)

    for name, metrics in combined_metrics.items():
        train_acc = metrics['final_train_acc']
        test_acc = metrics['final_test_acc']
        gap = train_acc - test_acc
        print(f"\n{name}:")
        print(f"  Training Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Accuracy Gap: {gap:.2f}%")
        if gap > 5:
            print(f"  -> Potential overfitting (gap > 5%)")
        else:
            print(f"  -> No significant overfitting")

    print("\n" + "="*60)
    print("OUTPUT FILES GENERATED:")
    print("="*60)
    import os
    files = ['training_curves_simple.png', 'training_curves_dropout.png',
             'sample_predictions_simple.png', 'sample_predictions_dropout.png',
             'cpu_vs_gpu_comparison.png', 'simple_fcnn.pth', 'fcnn_dropout.pth',
             'training_history.json', 'benchmark_results.txt', 'mnist_sample_images.png']

    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024  # KB
            print(f"  {f} ({size:.1f} KB)")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
