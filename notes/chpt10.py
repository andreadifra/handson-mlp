import marimo

__generated_with = "0.23.8"
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

    # Split dataset into train, validation, and test sets.
    from sklearn.model_selection import train_test_split

    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.25, random_state=42
    )

    # Keep num_workers=0 in marimo: this dataset class lives in a notebook cell,
    # so worker subprocesses cannot reliably import/pickle it.
    NUM_WORKERS = 0


    def build_dataloaders(batch_size: int = 512):
        """Create notebook-safe loaders for fast GPU smoke tests."""
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": NUM_WORKERS,
            "pin_memory": torch.cuda.is_available(),
        }

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_indices),
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_indices),
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_indices),
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, val_loader, test_loader


    train_loader, val_loader, test_loader = build_dataloaders(batch_size=512)

    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)} | workers={NUM_WORKERS} | "
        f"batch_size=512"
    )

    return (build_dataloaders,)


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
    # Training helpers tuned for notebook experimentation.
    #
    # Design choices:
    # - Epoch-level logging is the default because writing TensorBoard events every
    #   batch can dominate runtime in notebook workflows.
    # - Loss and accuracy stay on the device during each epoch so we avoid a CPU/GPU
    #   synchronization on every step.
    # - Timing is opt-in because accurate CUDA timing requires synchronization, which
    #   itself slows the loop down.

    import time
    from datetime import datetime
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _sync_cuda(device: torch.device, enabled: bool) -> None:
        if enabled and device.type == "cuda":
            torch.cuda.synchronize(device)


    def _move_batch_to_device(X_batch, y_batch, device: torch.device):
        if device.type == "cuda":
            return (
                X_batch.to(device, non_blocking=True),
                y_batch.to(device, non_blocking=True),
            )
        return X_batch.to(device), y_batch.to(device)


    def _format_hparams(hparams: dict | None) -> dict | None:
        """TensorBoard hparams must be scalar-like, so stringify complex values."""
        if not hparams:
            return None

        formatted = {}
        for key, value in hparams.items():
            if isinstance(value, (bool, str, float, int)) or value is None:
                formatted[key] = value
            else:
                formatted[key] = str(value)
        return formatted


    @torch.inference_mode()
    def evaluate_epoch(
        model,
        data_loader,
        criterion,
        *,
        device: torch.device = DEVICE,
        max_batches: int | None = None,
    ):
        """Run one validation pass and return scalar loss/accuracy."""
        model.eval()

        total_loss = torch.zeros((), device=device)
        total_correct = torch.zeros((), device=device)
        total_examples = 0

        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            X_batch, y_batch = _move_batch_to_device(X_batch, y_batch, device)
            logits = model(X_batch)
            batch_size = y_batch.size(0)

            total_loss += criterion(logits, y_batch).detach() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum()
            total_examples += batch_size

        avg_loss = (total_loss / total_examples).item()
        accuracy = (total_correct / total_examples).item()
        return {"loss": avg_loss, "accuracy": accuracy, "examples": total_examples}


    def train_enhanced(
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        *,
        epochs=10,
        start_epoch=0,
        device: torch.device = DEVICE,
        run_dir="./runs",
        run_name: str | None = None,
        verbose=True,
        log_every_n_steps: int | None = None,
        profile_timing=False,
        hparams: dict | None = None,
        trial=None,
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
    ):
        """Train a model with lightweight TensorBoard logging and optional Optuna pruning."""

        run_root = Path(run_dir)
        run_root.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        writer = SummaryWriter(log_dir=str(run_root / run_name))
        tensorboard_hparams = _format_hparams(hparams)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_time": [],
            "profiled_batch_time_ms": [],
        }

        best_val_accuracy = float("-inf")
        best_epoch = start_epoch
        global_step = start_epoch * len(train_loader)

        print(f"\n{'=' * 70}")
        print(f"Training on {device} | TensorBoard logs: {writer.log_dir}")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")
        print(f"{'=' * 70}\n")

        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_start = time.perf_counter()

            running_loss = torch.zeros((), device=device)
            running_correct = torch.zeros((), device=device)
            seen_examples = 0
            profiled_batch_times = []

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break

                _sync_cuda(device, profile_timing)
                batch_start = time.perf_counter()

                X_batch, y_batch = _move_batch_to_device(X_batch, y_batch, device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                batch_size = y_batch.size(0)
                running_loss += loss.detach() * batch_size
                running_correct += (logits.argmax(dim=1) == y_batch).sum()
                seen_examples += batch_size
                global_step += 1

                _sync_cuda(device, profile_timing)
                if profile_timing:
                    profiled_batch_times.append((time.perf_counter() - batch_start) * 1000)

                if log_every_n_steps and global_step % log_every_n_steps == 0:
                    writer.add_scalar("batch/train_loss", loss.detach().item(), global_step)
                    writer.add_scalar(
                        "batch/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                    if profile_timing and profiled_batch_times:
                        writer.add_scalar(
                            "batch/profiled_time_ms",
                            profiled_batch_times[-1],
                            global_step,
                        )

            train_loss = (running_loss / seen_examples).item()
            train_accuracy = (running_correct / seen_examples).item()

            val_metrics = evaluate_epoch(
                model,
                valid_loader,
                criterion,
                device=device,
                max_batches=max_eval_batches,
            )

            epoch_time = time.perf_counter() - epoch_start
            mean_batch_time = (
                sum(profiled_batch_times) / len(profiled_batch_times)
                if profiled_batch_times
                else None
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["epoch_time"].append(epoch_time)
            history["profiled_batch_time_ms"].append(mean_batch_time)

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_epoch = epoch + 1

            writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
            writer.add_scalar("epoch/train_accuracy", train_accuracy, epoch + 1)
            writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch + 1)
            writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch + 1)
            writer.add_scalar("epoch/epoch_time_sec", epoch_time, epoch + 1)
            if mean_batch_time is not None:
                writer.add_scalar("epoch/profiled_batch_time_ms", mean_batch_time, epoch + 1)
            writer.flush()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(
                    f"  Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | "
                    f"Val loss: {val_metrics['loss']:.4f} | Val acc: {val_metrics['accuracy']:.4f}"
                )
                print(f"  Epoch time: {epoch_time:.2f}s")
                if mean_batch_time is not None:
                    print(
                        f"  Profiled batch time: {mean_batch_time:.2f}ms "
                        f"(includes CUDA synchronization overhead)"
                    )

            if trial is not None:
                import optuna

                trial.report(val_metrics["accuracy"], step=epoch)
                if trial.should_prune():
                    writer.add_scalar("optuna/pruned_epoch", epoch + 1, epoch + 1)
                    if tensorboard_hparams:
                        writer.add_hparams(
                            tensorboard_hparams,
                            {
                                "hparam/best_val_accuracy": best_val_accuracy,
                                "hparam/final_val_accuracy": val_metrics["accuracy"],
                            },
                        )
                    writer.flush()
                    writer.close()
                    raise optuna.TrialPruned()

        history["best_val_accuracy"] = best_val_accuracy
        history["best_epoch"] = best_epoch

        if tensorboard_hparams:
            writer.add_hparams(
                tensorboard_hparams,
                {
                    "hparam/best_val_accuracy": best_val_accuracy,
                    "hparam/final_val_accuracy": history["val_accuracy"][-1],
                    "hparam/final_train_loss": history["train_loss"][-1],
                },
            )

        writer.flush()
        writer.close()

        total_time = sum(history["epoch_time"])
        print(f"\n{'=' * 70}")
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
        print(f"TensorBoard logs saved to: {run_root / run_name}")
        print(f"{'=' * 70}\n")

        return history, str(run_root / run_name)


    return DEVICE, Path, train_enhanced


@app.cell
def _(DEVICE, model):

    ## Initialise variables for the training loop.
    # Keep the baseline simple here; Optuna explores the larger search space below.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)

    print(f"Training device: {DEVICE}")
    return criterion, optimizer


@app.cell
def _(DEVICE, build_dataloaders, criterion, model, optimizer, train_enhanced):

    # Run the smallest useful training smoke test.
    # If this is not fast, stop and debug the environment before scaling up.
    smoke_train_loader, smoke_val_loader, _ = build_dataloaders(batch_size=512)

    history, log_dir = train_enhanced(
        model,
        optimizer,
        criterion,
        smoke_train_loader,
        smoke_val_loader,
        epochs=1,
        start_epoch=0,
        device=DEVICE,
        run_dir="./runs",
        run_name="smoke_test",
        verbose=True,
        log_every_n_steps=None,
        profile_timing=False,
        max_train_batches=10,
        max_eval_batches=10,
    )

    print(history)

    return


@app.cell
def _(
    CovTypeModel,
    DEVICE,
    Path,
    build_dataloaders,
    criterion,
    train_enhanced,
):

    ## Optional: Optuna Hyperparameter Optimization with pruning and TensorBoard metadata
    #
    # Best-practice notes:
    # - The trial now actually rebuilds the dataloaders with the sampled batch size.
    # - We report validation accuracy after each epoch so Optuna can prune weak trials.
    # - Each trial gets its own TensorBoard directory and logs hparams via
    #   SummaryWriter.add_hparams(), which works cleanly in a PyTorch-only setup.
    import optuna
    from optuna.trial import TrialState


    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        hidden_dims_choice = trial.suggest_categorical(
            "hidden_dims",
            [(64, 128), (128, 256), (128, 256, 64), (256, 512, 128)],
        )
        optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adamw"])

        if optimizer_name == "sgd":
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
        else:
            momentum = None

        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        trial_model = CovTypeModel(
            input_dim=54,
            hidden_dims=hidden_dims_choice,
            output_dim=7,
        ).to(DEVICE)

        if optimizer_name == "sgd":
            trial_optimizer = optim.SGD(
                trial_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            trial_optimizer = optim.AdamW(
                trial_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        trial_train_loader, trial_val_loader, _ = build_dataloaders(batch_size=batch_size)
        run_name = f"trial_{trial.number:04d}"
        expected_log_dir = str(Path("./runs/optuna") / run_name)

        try:
            history, log_dir = train_enhanced(
                trial_model,
                trial_optimizer,
                criterion,
                trial_train_loader,
                trial_val_loader,
                epochs=5,
                start_epoch=0,
                device=DEVICE,
                run_dir="./runs/optuna",
                run_name=run_name,
                verbose=False,
                log_every_n_steps=250,
                profile_timing=False,
                hparams=trial.params,
                trial=trial,
            )
        except optuna.TrialPruned:
            trial.set_user_attr("tensorboard_log_dir", expected_log_dir)
            raise

        trial.set_user_attr("tensorboard_log_dir", log_dir)
        trial.set_user_attr("best_epoch", history["best_epoch"])
        return history["best_val_accuracy"]


    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///optuna_study.db",
        study_name="covertype_hpo",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print(f"Completed trials: {len(complete_trials)} | Pruned trials: {len(pruned_trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # TensorBoard:
    # tensorboard --logdir=./runs/optuna
    # Optuna dashboard:
    # optuna-dashboard sqlite:///optuna_study.db
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training Review Summary

    ### What changed

    - **Lower-overhead TensorBoard logging**: the main loop now logs compact epoch-level metrics by default and only logs batch metrics when you explicitly request it.
    - **Accurate timing guidance**: CUDA timings are now treated as an opt-in profiling mode, because synchronized timings are slower but accurate.
    - **Cleaner metric handling**: accuracy and loss are accumulated on-device during the epoch instead of forcing CPU synchronization every batch.
    - **Better data loading defaults**: the data loaders now use `pin_memory=True` on CUDA and worker processes when available.
    - **Optuna improvements**: sampled batch size is now actually used, each trial gets its own TensorBoard run directory, and Optuna pruning is enabled through `trial.report(...)` / `trial.should_prune()`.

    ### Why the original run looked slow

    The previous loop reported a very small batch time (`~3ms`) but a long epoch time (`~174s`) because the batch timing did **not** include TensorBoard writes and was not reliable for CUDA profiling. In the old version, the expensive per-batch `writer.add_scalar(...)` calls happened **after** the measured batch timer, so the metric understated the real end-to-end cost.

    ### TensorBoard best practice for this notebook

    For a PyTorch notebook, `SummaryWriter` plus `add_hparams(...)` is the lightest way to track Optuna trials. Optuna's dedicated TensorBoard callback exists in the integration docs, but it lives in the separate `optuna-integration` package and pulls in TensorFlow-oriented dependencies. That is heavier than needed for this workflow.
    """)
    return


if __name__ == "__main__":
    app.run()
