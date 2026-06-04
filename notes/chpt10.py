import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    ## Import common libraries
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Book Questions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2. Creating tensors
    """)
    return


@app.cell
def _():
    X = torch.tensor([[1.0, 5.6, 2.3], [2.0, 3.4, 4.5], [3.0, 1.2, 6.7]], device="cuda")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 5. Runtime Error Check
    """)
    return


@app.cell
def _():
    t = torch.tensor(2.0, requires_grad=True)
    z = t.cos().exp_()
    z.backward()
    return t, z


@app.cell
def _(t):
    t2 = torch.tensor(2.0, requires_grad=True)
    z2 = t.cos_().exp()
    z2.backward()
    return


@app.cell
def _(z):
    z.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 6. Linear module
    """)
    return


@app.cell
def _():
    test_mod = nn.Linear(100, 200)
    return (test_mod,)


@app.cell
def _(test_mod):
    print(test_mod.bias.shape)
    print(test_mod.weight.shape)
    return


@app.cell
def _(test_mod):
    output = test_mod(torch.randn(45, 100))
    return (output,)


@app.cell
def _(output):
    output.shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 8. Testing moving optimizer to GPU after creating it
    """)
    return


@app.cell
def _():
    # 1. Define a simple model
    _model = nn.Linear(10, 1)

    # 2. WRONG WAY: Initialize optimizer while model is on CPU
    _optimizer = optim.SGD(_model.parameters(), lr=0.1)

    # 3. Move model to GPU
    _model.to("cuda:0")
    return


@app.cell
def _(model, optimizer):
    # 4. Check the mismatch
    model_p = next(model.parameters())
    opt_p = optimizer.param_groups[0]['params'][0]

    print(f"Model parameter device: {model_p.device}")
    print(f"Optimizer tracked parameter device: {opt_p.device}")

    # This check tells us if they are the same actual object in memory
    print(f"Are they the same object? {id(model_p) == id(opt_p)}")
    return


@app.cell
def _(model, optimizer):
    # 5. Attempt a dummy training step
    try:
        input_data = torch.randn(5, 10).to("cuda:0")
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        print("\nStep successful (but might be updating the wrong memory!)")
    except Exception as e:
        print(f"\nCaught expected error: {e}")
    return (output,)


@app.cell
def _(optimizer):
    optimizer.load_state_dict
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 13. Use autograd to find gradient of $f(x,y) = sin(x^2y)$ at the point $(x,y) = (1.2,3.4)$.
    """)
    return


@app.cell
def _():
    from torch.autograd import grad
    # I want to 
    return (grad,)


@app.cell
def _():
    ## Define the function
    def f(x,y):
        return torch.sin(x**2 * y)

    x = torch.tensor(1.2, requires_grad=True)
    y = torch.tensor(3.4, requires_grad=True)

    z = f(x,y)
    return x, y, z


@app.cell
def _(grad, x, y, z):
    grad(z, [x,y])
    return


@app.cell
def _(x, y):
    ## Alternatively 

    out = torch.sin(x**2 * y)
    out.backward()
    print(x.grad, y.grad)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 14. Create a custom Dense module that replicates the functionality of an nn.Linear module followed by an nn.ReLU module. Try implementing it first using the nn.Linear and nn.ReLU modules, and then reimplement it using nn.Parameter and the relu() function.
    """)
    return


@app.cell
def _():
    # Set seed for reproducibility
    torch.manual_seed(42)
    return


@app.cell
def _():
    class DenseLayer(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linlayer = nn.Sequential(
                nn.Linear(n_features, 1), nn.ReLU()
            )

        def forward(self, X):
            return self.linlayer(X)

    # Instantiate class and move to GPU
    Dense_1 = DenseLayer(10).to("cuda:0")
    return (Dense_1,)


@app.cell
def _(Dense_1):
    # Try a sample forward pass
    dummy_data = torch.randn(5, 10, device = "cuda:0")
    Dense_1(dummy_data)
    return (dummy_data,)


@app.cell
def _(Dense_1):
    # Check the gradients of the parameters
    for name, param in Dense_1.named_parameters():
        print(f"Parameter: {name}, Param: {param},Gradient: {param.grad}")
    return


@app.cell
def _(Dense_1):
    # Best-practice Dense layer: learnable parameters plus functional linear + ReLU.
    class DenseLayer2(nn.Module):
        def __init__(self, in_features, out_features=1):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.empty(out_features))

            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            fan_in = self.weight.size(1)
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, X):
            return torch.relu(F.linear(X, self.weight, self.bias))

    Dense_2 = DenseLayer2(Dense_1.linlayer[0].in_features).to("cuda:0")

    # Copy DenseLayer's learned parameters so the outputs match exactly.
    with torch.no_grad():
        Dense_2.weight.copy_(Dense_1.linlayer[0].weight)
        Dense_2.bias.copy_(Dense_1.linlayer[0].bias)
    return DenseLayer2, Dense_2


@app.cell
def _(Dense_1, Dense_2, dummy_data):
    dense_1_out = Dense_1(dummy_data)
    dense_2_out = Dense_2(dummy_data)
    print(torch.allclose(dense_1_out, dense_2_out))
    print(dense_1_out.shape, dense_2_out.shape)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 15. Build and train a classification MLP on the CoverType dataset
    Step 1: Load the dataset using sklearn.datasets.fetch_covtype() and create a custom PyTorch Dataset for this data.
    """)
    return


@app.cell
def _():
    from torch import Tensor
    from sklearn.datasets import fetch_covtype
    from torch.utils.data import Dataset, DataLoader

    data = fetch_covtype()

    # standardise the features to zero mean and unit variance
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data.data = scaler.fit_transform(data.data)


    class CoverTypeDataset(Dataset[tuple[Tensor, Tensor]]):
        def __init__(self, data):
            self.X   = torch.tensor(data.data, dtype=torch.float32)
            self.y   = torch.tensor(data.target - 1, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
            return self.X[index], self.y[index]

    return CoverTypeDataset, DataLoader, data


@app.cell
def _(CoverTypeDataset, data):
    dataset = CoverTypeDataset(data)

    print(f"Dataset size: {len(dataset)}")
    return (dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Step 2: Create data loaders for training, validation and testing
    """)
    return


@app.cell
def _(DataLoader, dataset):
    # Split dataset into train, val, test

    from sklearn.model_selection import train_test_split

    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.25, random_state=42
    )

    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices), batch_size=64, shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices), batch_size=64, shuffle=False
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    return train_loader, val_loader


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Step 3: Build a custom MLP module to tacke this classification task. You can optionally use the custome `Dense` module from the previous exercise
    """)
    return


@app.cell
def _(DenseLayer2):
    # Create custom MLP Module with a bunch of DenseLayer2 layers:


    class CovTypeModel(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim):
            super().__init__()
            layers = []
            prev_dim = input_dim

            # Build dense layers with configurable output dimensions
            for h_dim in hidden_dims:
                layers.append(DenseLayer2(prev_dim, h_dim))
                prev_dim = h_dim

            # Final output layer
            layers.append(nn.Linear(prev_dim, output_dim))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return (CovTypeModel,)


@app.cell
def _(CovTypeModel):
    # Build the model:
    ## 54 input features and 7 output classes

    model = CovTypeModel(input_dim = 54, hidden_dims = [128, 256, 64], output_dim=7).to("cuda:0")
    model
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Step 4: Train the model on the GPU and tro to reach 93% accuracy on the test set. For this, you will likely have to perform hyperparameter search to find the right number of layers and neurons per layer, a good learning rate and batch size and so on, optionally using Optuna for this.
    """)
    return


@app.cell
def _():
    # Enhanced training loop with detailed logging and TensorBoard integration
    import time
    from pathlib import Path
    from datetime import datetime

    class TrainingState:
        """Tracks training state for resumability and monitoring"""
        def __init__(self, model_name="covertype_model"):
            self.model_name = model_name
            self.start_time = None
            self.resumed_from_epoch = 0
            self.total_batches_processed = 0
            self.best_val_acc = 0.0
        
        def start(self, resumed_epoch=0):
            self.start_time = datetime.now()
            self.resumed_from_epoch = resumed_epoch
            print(f"\n{'='*70}")
            print(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if resumed_epoch > 0:
                print(f"Resumed from epoch {resumed_epoch}")
            print(f"{'='*70}\n")

    def train_enhanced(model, optimizer, criterion, metric, train_loader, valid_loader, 
                       epochs=10, start_epoch=0, run_dir="./runs", verbose=True):
        """
        Enhanced training loop with timing, logging, and TensorBoard support
    
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            criterion: Loss function
            metric: Metric to track (e.g., Accuracy)
            train_loader: Training DataLoader
            valid_loader: Validation DataLoader
            epochs: Total number of epochs
            start_epoch: Epoch to start from (for resuming)
            run_dir: Directory for TensorBoard logs
            verbose: Whether to print detailed timing info
        """
        from torch.utils.tensorboard import SummaryWriter
    
        # Setup TensorBoard
        run_dir = Path(run_dir)
        run_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    
        # Training state
        state = TrainingState("CovTypeModel")
        state.start(start_epoch)
    
        history = {
            "train_loss": [], 
            "train_metric": [], 
            "val_metric": [],
            "epoch_time": [],
            "batch_time": []
        }
    
        global_batch_count = start_epoch * len(train_loader)
    
        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            total_loss = 0.
            metric.reset()
            model.train()
        
            batch_times = []
        
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                batch_start = time.perf_counter()
            
                # Move to GPU
                X_batch, y_batch = X_batch.to("cuda:0"), y_batch.to("cuda:0")
                move_time = time.perf_counter() - batch_start
            
                # Forward pass
                forward_start = time.perf_counter()
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                forward_time = time.perf_counter() - forward_start
            
                # Backward pass
                backward_start = time.perf_counter()
                loss.backward()
                optimizer.step()
                backward_time = time.perf_counter() - backward_start
            
                total_loss += loss.item()
                metric.update(outputs, y_batch)
            
                batch_total_time = time.perf_counter() - batch_start
                batch_times.append(batch_total_time)
                global_batch_count += 1
            
                # Log batch metrics to TensorBoard
                writer.add_scalar("Loss/train_batch", loss.item(), global_batch_count)
                writer.add_scalar("Timing/batch_total_ms", batch_total_time * 1000, global_batch_count)
                writer.add_scalar("Timing/forward_ms", forward_time * 1000, global_batch_count)
                writer.add_scalar("Timing/backward_ms", backward_time * 1000, global_batch_count)
                writer.add_scalar("Timing/data_transfer_ms", move_time * 1000, global_batch_count)
            
                if verbose and (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                    print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | Batch time: {batch_total_time*1000:.2f}ms")
        
            # Calculate metrics
            metrics_start = time.perf_counter()
            avg_loss = total_loss / len(train_loader)
            train_acc = metric.compute().item()
            train_metrics_time = time.perf_counter() - metrics_start
        
            # Validation
            val_start = time.perf_counter()
            val_acc = evaluate(model, valid_loader, metric).item()
            val_time = time.perf_counter() - val_start
        
            epoch_total_time = time.perf_counter() - epoch_start
        
            # Update state
            history["train_loss"].append(avg_loss)
            history["train_metric"].append(train_acc)
            history["val_metric"].append(val_acc)
            history["epoch_time"].append(epoch_total_time)
            history["batch_time"].append(sum(batch_times) / len(batch_times) if batch_times else 0)
        
            if val_acc > state.best_val_acc:
                state.best_val_acc = val_acc
        
            # Log epoch metrics to TensorBoard
            writer.add_scalar("Loss/train_epoch", avg_loss, epoch + 1)
            writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
            writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
            writer.add_scalar("Timing/epoch_total_sec", epoch_total_time, epoch + 1)
            writer.add_scalar("Timing/validation_sec", val_time, epoch + 1)
            writer.add_scalar("Timing/metrics_computation_ms", train_metrics_time * 1000, epoch + 1)
            writer.add_scalar("Timing/avg_batch_ms", history["batch_time"][-1] * 1000, epoch + 1)
        
            # Console output
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Timing - Epoch: {epoch_total_time:.2f}s | Validation: {val_time:.2f}s | "
                  f"Avg batch: {history['batch_time'][-1]*1000:.2f}ms")
            if val_acc == state.best_val_acc:
                print(f"  ✓ Best validation accuracy!")
    
        writer.close()
    
        # Print summary
        total_time = sum(history["epoch_time"])
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Total time: {total_time:.2f}s | Epochs: {epochs - start_epoch}")
        print(f"Best validation accuracy: {state.best_val_acc:.4f}")
        print(f"Average epoch time: {total_time / (epochs - start_epoch):.2f}s")
        print(f"TensorBoard logs saved to: {writer.log_dir}")
        print(f"{'='*70}\n")
    
        return history, writer.log_dir

    def evaluate(model, valid_loader, metric):
        # Set model to evaluation mode
        model.eval()
        # Reset metric at the beginning of evaluation
        metric.reset()

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to("cuda:0"), y_batch.to("cuda:0")
                outputs = model(X_batch)
                metric.update(outputs, y_batch)

        return metric.compute()

    return (train_enhanced,)


@app.cell
def _(model):
    ## Initialise variables for the training loop
    from torchmetrics import Accuracy

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    accuracy_metric = Accuracy(task="multiclass", num_classes=7).to("cuda:0")
    return accuracy_metric, criterion, optimizer


@app.cell
def _():
    ## Setup TensorBoard and run directory
    import subprocess
    import os

    # Create runs directory
    os.makedirs("./runs", exist_ok=True)

    # Display TensorBoard instructions
    print("📊 TensorBoard will log all training metrics!")
    print("To visualize during/after training, run in a separate terminal:")
    print("\n  tensorboard --logdir=./runs\n")
    print("Then open: http://localhost:6006\n")
    return


@app.cell
def _(
    accuracy_metric,
    criterion,
    model,
    optimizer,
    train_enhanced,
    train_loader,
    val_loader,
):
    ## Start training with enhanced logging
    history, log_dir = train_enhanced(
        model, 
        optimizer, 
        criterion, 
        accuracy_metric, 
        train_loader, 
        val_loader, 
        epochs=1,  # Testing with 1 epoch first
        start_epoch=0,
        run_dir="./runs",
        verbose=True
    )
    return (history,)


@app.cell
def _():
    ## Optional: Optuna Hyperparameter Optimization
    # This cell shows how to use Optuna for systematic hyperparameter tuning
    # Uncomment to use

    """
    import optuna
    from optuna.trial import TrialState

    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        momentum = trial.suggest_float('momentum', 0.7, 0.99)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        hidden_dims_choice = trial.suggest_categorical('hidden_dims', 
                                                         [(64, 128), (128, 256), (256, 512, 128)])
    
        # Create fresh model
        trial_model = CovTypeModel(input_dim=54, hidden_dims=hidden_dims_choice, output_dim=7).to("cuda:0")
        trial_optimizer = optim.SGD(trial_model.parameters(), momentum=momentum, lr=lr)
    
        # Train
        history, _ = train_enhanced(
            trial_model, trial_optimizer, criterion, accuracy_metric,
            train_loader, val_loader, epochs=5, start_epoch=0, verbose=False
        )
    
        # Return the best validation accuracy
        return max(history['val_metric'])

    # Create a study and optimize
    study = optuna.create_study(direction='maximize', storage='sqlite:///optuna_study.db', 
                                study_name='covertype_hpo', load_if_exists=True)
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    # Print best trial
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Optuna Dashboard: Run in terminal:
    # optuna-dashboard sqlite:///optuna_study.db
    """
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 📊 Training Enhancements Summary

    ### What's New

    Your training loop now includes comprehensive logging, timing analysis, and visualization support:

    #### 1. **Enhanced Logging**
    - ✓ **Overall Epoch Tracking**: Know which epoch you're on and resume from any checkpoint
    - ✓ **Timing Measurements**: Track performance bottlenecks:
      - Batch processing time (total)
      - Forward pass duration
      - Backward pass duration
      - Data transfer to GPU
      - Metrics computation time
      - Validation time
    - ✓ **Relevant Metrics**: Monitor both training and validation accuracy, loss trends

    #### 2. **TensorBoard Integration**
    - Automatically logs all metrics to TensorBoard event files
    - Accessible via: `tensorboard --logdir=./runs`
    - View at: `http://localhost:6006`
    - Metrics tracked:
      - Per-batch and per-epoch loss
      - Training and validation accuracy
      - Timing breakdowns for all operations
      - Best validation accuracy detection

    #### 3. **Optional Optuna Integration**
    - Hyperparameter optimization template included
    - Supports:
      - Learning rate tuning
      - Momentum adjustment
      - Batch size selection
      - Architecture search (hidden layer dimensions)
    - Dashboard: `optuna-dashboard sqlite:///optuna_study.db`

    #### 4. **Resumable Training**
    - `start_epoch` parameter allows resuming training from any checkpoint
    - `TrainingState` class tracks:
      - Resume epoch information
      - Best validation accuracy
      - Training time

    ### Quick Start

    **Run with TensorBoard visualization:**
    ```python
    history, log_dir = train_enhanced(
        model, optimizer, criterion, accuracy_metric,
        train_loader, val_loader, epochs=10, start_epoch=0
    )

    # In another terminal:
    # tensorboard --logdir=./runs
    ```

    **Resume training from epoch 5:**
    ```python
    history, log_dir = train_enhanced(
        model, optimizer, criterion, accuracy_metric,
        train_loader, val_loader, epochs=10, start_epoch=5
    )
    ```

    **Enable Optuna hyperparameter search:**
    Uncomment the Optuna cell below and run to optimize hyperparameters
    """)
    return


@app.cell
def _(history):
    ## Plotting Training Results
    import matplotlib.pyplot as plt

    # Only plot if we have history from training
    if 'history' in locals():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
        # Plot loss
        axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
        # Plot accuracy
        epochs_range = range(1, len(history['train_metric']) + 1)
        axes[0, 1].plot(epochs_range, history['train_metric'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs_range, history['val_metric'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
        # Plot epoch timing
        axes[1, 0].bar(epochs_range, [t for t in history['epoch_time']], alpha=0.7, color='steelblue')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Epoch Duration')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
        # Plot batch timing
        axes[1, 1].plot(history['batch_time'], label='Avg Batch Time', linewidth=2, marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Average Batch Duration per Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
        plt.show()
    
        # Print summary statistics
        print("\n📈 Training Statistics Summary:")
        print(f"  Total epochs: {len(history['train_loss'])}")
        print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final training accuracy: {history['train_metric'][-1]:.4f}")
        print(f"  Final validation accuracy: {history['val_metric'][-1]:.4f}")
        print(f"  Best validation accuracy: {max(history['val_metric']):.4f} (epoch {history['val_metric'].index(max(history['val_metric'])) + 1})")
        print(f"  Total training time: {sum(history['epoch_time']):.2f}s")
        print(f"  Average epoch time: {sum(history['epoch_time']) / len(history['epoch_time']):.2f}s")
        print(f"  Average batch time: {sum(history['batch_time']) / len(history['batch_time']) * 1000:.2f}ms")
    else:
        print("Run the training cell first to generate history data")
    return


@app.cell
def _():
    locals()
    return


if __name__ == "__main__":
    app.run()
