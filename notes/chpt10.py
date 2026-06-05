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
    X = torch.tensor(
        [[1.0, 5.6, 2.3], [2.0, 3.4, 4.5], [3.0, 1.2, 6.7]], device="cuda"
    )
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
    opt_p = optimizer.param_groups[0]["params"][0]

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
    def f(x, y):
        return torch.sin(x**2 * y)


    x = torch.tensor(1.2, requires_grad=True)
    y = torch.tensor(3.4, requires_grad=True)

    z = f(x, y)
    return x, y, z


@app.cell
def _(grad, x, y, z):
    grad(z, [x, y])
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
            self.linlayer = nn.Sequential(nn.Linear(n_features, 1), nn.ReLU())

        def forward(self, X):
            return self.linlayer(X)


    # Instantiate class and move to GPU
    Dense_1 = DenseLayer(10).to("cuda:0")
    return (Dense_1,)


@app.cell
def _(Dense_1):
    # Try a sample forward pass
    dummy_data = torch.randn(5, 10, device="cuda:0")
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

            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            fan_in = self.weight.size(1)
            bound = 1 / fan_in**0.5
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
            self.X = torch.tensor(data.data, dtype=torch.float32)
            self.y = torch.tensor(data.target - 1, dtype=torch.long)

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

    # Split the dataset once so every experiment uses the same partition.
    from sklearn.model_selection import train_test_split

    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.25, random_state=42
    )

    # Build loaders from a TensorDataset rather than the notebook-local custom
    # Dataset class. This avoids the `spawn` pickling error that can happen when a
    # worker process tries to import a class defined inside a notebook cell.
    #
    # We still keep `num_workers=0` here because the feature tensors are already in
    # memory, so extra worker processes only add overhead in this notebook.
    base_dataset = torch.utils.data.TensorDataset(dataset.X, dataset.y)
    DEFAULT_BATCH_SIZE = 512


    def build_dataloaders(batch_size: int = DEFAULT_BATCH_SIZE):
        """Return train/validation/test loaders for the CoverType tensors.

        Parameters
        ----------
        batch_size:
            Number of examples per batch. This is worth tuning because it changes
            both training speed and optimization behaviour.

        Notes
        -----
        `num_workers` is intentionally fixed at zero. In this notebook the data is
        already held in memory as tensors, so multiprocessing does not help. The
        earlier failure was not that marimo forbids workers; it was that the custom
        dataset class lived inside a notebook cell and could not be reconstructed by
        spawned worker processes.
        """
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        }

        train_loader = DataLoader(
            torch.utils.data.Subset(base_dataset, train_indices),
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(base_dataset, val_indices),
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(base_dataset, test_indices),
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, val_loader, test_loader


    train_loader, val_loader, test_loader = build_dataloaders()

    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)} | batch_size={DEFAULT_BATCH_SIZE}"
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

    model = CovTypeModel(
        input_dim=54, hidden_dims=[128, 256, 64], output_dim=7
    ).to("cuda:0")
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

    # Compact training helpers for notebook use.
    #
    # The previous version grew too complicated and made it harder to see the real
    # bottlenecks. This version keeps only the pieces we actually need:
    # - move batches onto the current device
    # - evaluate a model on validation data
    # - train for a few epochs
    # - optionally report validation accuracy to Optuna for pruning
    import time
    from datetime import datetime
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def move_batch_to_device(X_batch, y_batch, device: torch.device):
        """Move one batch onto the selected device.

        When CUDA is available we request a non-blocking transfer. This only helps
        when the DataLoader uses pinned memory, but it is a harmless default and it
        keeps the function usable for both CPU and GPU runs.
        """
        if device.type == "cuda":
            return (
                X_batch.to(device, non_blocking=True),
                y_batch.to(device, non_blocking=True),
            )
        return X_batch.to(device), y_batch.to(device)


    @torch.inference_mode()
    def evaluate_model(
        model,
        data_loader,
        criterion,
        *,
        device: torch.device = DEVICE,
        max_batches: int | None = None,
    ):
        """Evaluate the model on a validation loader.

        Parameters
        ----------
        model:
            The model to evaluate.
        data_loader:
            Validation or test loader.
        criterion:
            Loss function used for reporting validation loss.
        device:
            Target device.
        max_batches:
            Optional cap used for smoke tests. This keeps notebook checks fast.

        Returns
        -------
        dict
            A dictionary with average loss, accuracy, and example count.
        """
        model.eval()

        total_loss = torch.zeros((), device=device)
        total_correct = torch.zeros((), device=device)
        total_examples = 0

        for batch_index, (X_batch, y_batch) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            X_batch, y_batch = move_batch_to_device(X_batch, y_batch, device)
            logits = model(X_batch)
            batch_size = y_batch.size(0)

            total_loss += criterion(logits, y_batch).detach() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum()
            total_examples += batch_size

        return {
            "loss": (total_loss / total_examples).item(),
            "accuracy": (total_correct / total_examples).item(),
            "examples": total_examples,
        }


    def train_model(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        *,
        epochs: int = 10,
        device: torch.device = DEVICE,
        run_dir: str = "./runs",
        run_name: str | None = None,
        verbose: bool = True,
        trial=None,
        hparams: dict | None = None,
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
    ):
        """Train a model and log epoch metrics to TensorBoard.

        This function is intentionally small and notebook-friendly. It logs only
        epoch-level metrics because per-batch TensorBoard writes can dominate
        runtime when many short batches are processed.

        Parameters
        ----------
        model, optimizer, criterion:
            Standard PyTorch training objects.
        train_loader, val_loader:
            DataLoaders for training and validation.
        epochs:
            Number of full passes over the training data.
        device:
            Training device.
        run_dir, run_name:
            TensorBoard output location.
        verbose:
            Whether to print epoch summaries.
        trial:
            Optional Optuna trial. When provided, validation accuracy is reported to
            Optuna after each epoch so weak trials can be pruned early.
        hparams:
            Optional hyperparameters to record in TensorBoard.
        max_train_batches, max_eval_batches:
            Optional batch caps used for smoke tests and tiny examples.

        Returns
        -------
        tuple[dict, str]
            Training history and the TensorBoard log directory.
        """
        run_root = Path(run_dir)
        run_root.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_dir = str(run_root / run_name)
        writer = SummaryWriter(log_dir=log_dir)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_time": [],
            "best_val_accuracy": 0.0,
            "best_epoch": 0,
        }

        for epoch in range(epochs):
            model.train()
            epoch_start = time.perf_counter()

            running_loss = torch.zeros((), device=device)
            running_correct = torch.zeros((), device=device)
            seen_examples = 0

            for batch_index, (X_batch, y_batch) in enumerate(train_loader):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break

                X_batch, y_batch = move_batch_to_device(X_batch, y_batch, device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                batch_size = y_batch.size(0)
                running_loss += loss.detach() * batch_size
                running_correct += (logits.argmax(dim=1) == y_batch).sum()
                seen_examples += batch_size

            train_loss = (running_loss / seen_examples).item()
            train_accuracy = (running_correct / seen_examples).item()
            val_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
                device=device,
                max_batches=max_eval_batches,
            )
            epoch_time = time.perf_counter() - epoch_start

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["epoch_time"].append(epoch_time)

            if val_metrics["accuracy"] > history["best_val_accuracy"]:
                history["best_val_accuracy"] = val_metrics["accuracy"]
                history["best_epoch"] = epoch + 1

            writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
            writer.add_scalar("epoch/train_accuracy", train_accuracy, epoch + 1)
            writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch + 1)
            writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch + 1)
            writer.add_scalar("epoch/time_sec", epoch_time, epoch + 1)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(
                    f"  Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | "
                    f"Val loss: {val_metrics['loss']:.4f} | Val acc: {val_metrics['accuracy']:.4f}"
                )
                print(f"  Epoch time: {epoch_time:.2f}s")

            if trial is not None:
                import optuna

                trial.report(val_metrics["accuracy"], step=epoch)
                if trial.should_prune():
                    writer.close()
                    raise optuna.TrialPruned()

        # Record hyperparameters at the end of the run. TensorBoard expects scalar
        # values, so we stringify non-scalar objects such as hidden-dimension tuples.
        if hparams:
            writer.add_hparams(
                {
                    key: value if isinstance(value, (bool, int, float, str)) else str(value)
                    for key, value in hparams.items()
                },
                {
                    "hparam/best_val_accuracy": history["best_val_accuracy"],
                    "hparam/final_val_accuracy": history["val_accuracy"][-1],
                },
            )

        writer.close()
        return history, log_dir


    return DEVICE, train_model


@app.cell
def _(DEVICE, model):

    # Baseline training objects.
    #
    # We keep the optimizer fixed to SGD with momentum=0.9. That makes the Optuna
    # search easier to reason about because only learning rate, batch size, and the
    # hidden-layer layout are changing.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)

    print(f"Training device: {DEVICE}")
    return criterion, optimizer


@app.cell
def _(DEVICE, build_dataloaders, criterion, model, optimizer, train_model):

    # Minimal smoke test.
    #
    # This is intentionally tiny: one epoch, five training batches, and five
    # validation batches. If even this takes a long time, the problem is with the
    # environment or data path rather than the model architecture.
    smoke_train_loader, smoke_val_loader, _ = build_dataloaders(batch_size=512)

    history, log_dir = train_model(
        model,
        optimizer,
        criterion,
        smoke_train_loader,
        smoke_val_loader,
        epochs=1,
        device=DEVICE,
        run_dir="./runs",
        run_name="smoke_test",
        verbose=True,
        max_train_batches=5,
        max_eval_batches=5,
    )

    print(history)
    return


@app.cell
def _(CovTypeModel, DEVICE, build_dataloaders, criterion, train_model):
    ## Optional: Optuna search helpers.
    import optuna

    # Optuna stores parameter choices inside the study database. The earlier version
    # used tuples/lists directly for `hidden_dims`, which caused reload mismatches in
    # persisted studies. We therefore encode each architecture as a stable string
    # key and decode it inside the objective function.
    HIDDEN_DIM_OPTIONS = {
        "64-128": (64, 128),
        "128-256": (128, 256),
        "128-256-64": (128, 256, 64),
        "256-512-128": (256, 512, 128),
    }

    DEFAULT_STUDY_NAME = "covertype_hpo_v2"


    def make_objective(
        *,
        epochs: int = 5,
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
    ):
        """Create an Optuna objective for the CoverType MLP.

        Parameters
        ----------
        epochs:
            Number of epochs per trial.
        max_train_batches, max_eval_batches:
            Optional caps for very small smoke tests. Leave these as `None` for a
            real study.
        """

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
            batch_size = trial.suggest_categorical(
                "batch_size", [128, 256, 512, 1024]
            )
            hidden_dims_key = trial.suggest_categorical(
                "hidden_dims_key",
                list(HIDDEN_DIM_OPTIONS.keys()),
            )
            hidden_dims = HIDDEN_DIM_OPTIONS[hidden_dims_key]

            trial_model = CovTypeModel(
                input_dim=54,
                hidden_dims=hidden_dims,
                output_dim=7,
            ).to(DEVICE)
            trial_optimizer = optim.SGD(
                trial_model.parameters(),
                lr=lr,
                momentum=0.9,
            )
            trial_train_loader, trial_val_loader, _ = build_dataloaders(
                batch_size=batch_size
            )

            history, log_dir = train_model(
                trial_model,
                trial_optimizer,
                criterion,
                trial_train_loader,
                trial_val_loader,
                epochs=epochs,
                device=DEVICE,
                run_dir="./runs/optuna",
                run_name=f"trial_{trial.number:04d}",
                verbose=False,
                trial=trial,
                hparams={
                    "lr": lr,
                    "batch_size": batch_size,
                    "hidden_dims_key": hidden_dims_key,
                    "hidden_dims": hidden_dims,
                },
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
            )

            trial.set_user_attr("tensorboard_log_dir", log_dir)
            trial.set_user_attr("hidden_dims", list(hidden_dims))
            trial.set_user_attr("best_epoch", history["best_epoch"])
            return history["best_val_accuracy"]

        return objective


    def create_optuna_study(
        *,
        study_name: str = DEFAULT_STUDY_NAME,
        storage: str = "sqlite:///optuna_study.db",
    ):
        """Create or reload the CoverType Optuna study.

        Notes
        -----
        We default to `covertype_hpo_v2` instead of the earlier study name because
        Optuna persists the search-space schema. The old study used an incompatible
        representation for the architecture parameter, which is why
        `study.optimize(...)` later raised a `ValueError` during reload.
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        return optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )


    def run_optuna_search(
        *,
        n_trials: int = 10,
        epochs: int = 5,
        study_name: str = DEFAULT_STUDY_NAME,
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
    ):
        """Run the Optuna search and return the completed study.

        This helper is separate from the definition cell on purpose. It prevents the
        notebook from launching a full study every time an upstream cell changes.
        """
        study = create_optuna_study(study_name=study_name)
        study.optimize(
            make_objective(
                epochs=epochs,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        return study


    print(
        "Optuna helpers ready. Example: "
        "run_optuna_search(n_trials=1, epochs=1, max_train_batches=5, max_eval_batches=5, study_name='covertype_hpo_smoke')"
    )
    return (run_optuna_search,)


@app.cell
def _(run_optuna_search):
    run_optuna_search()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training Notes

    - The training utilities have been simplified so the notebook is easier to read and debug.
    - `build_dataloaders()` keeps `num_workers=0` by design because the data already lives in memory as tensors and multiprocessing was slower here.
    - The Optuna section now searches only `lr`, `batch_size`, and `hidden_dims` while keeping `SGD(momentum=0.9)` fixed.
    - The earlier Optuna `ValueError` came from reusing a persisted study whose old architecture parameter encoding no longer matched the new code. The notebook now uses stable string keys such as `128-256-64` and a new default study name.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
