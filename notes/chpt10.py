import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import contextlib
    import marimo as mo
    import numpy as np
    import optuna
    import time

    from datetime import datetime
    from pathlib import Path

    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from threading import Event

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch import Tensor
    from torch.autograd import grad
    from torch.profiler import ProfilerActivity, profile, record_function, schedule
    from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    # Shared device used throughout the notebook. Cells should use this instead of
    # hard-coding CUDA strings so the notebook still runs on CPU-only machines.
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        [[1.0, 5.6, 2.3], [2.0, 3.4, 4.5], [3.0, 1.2, 6.7]],
        device=DEFAULT_DEVICE,
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
    return (z,)


@app.cell
def _():
    t2 = torch.tensor(2.0, requires_grad=True)
    z2 = t2.cos_().exp()
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
    # 1. Define a simple model for the optimizer/device demo.
    #
    # This cell intentionally keeps the entire experiment together so the notebook
    # reads sequentially: create the module, create the optimizer, move the module,
    # inspect the parameter references, and run one step. Splitting these tiny
    # fragments across multiple active cells made the section harder to follow.
    demo_model = nn.Linear(10, 1)

    # 2. Initialize the optimizer before moving the model. The goal is to inspect
    # whether moving the model afterwards invalidates the optimizer's parameter
    # references.
    demo_optimizer = optim.SGD(demo_model.parameters(), lr=0.1)

    # 3. Move the model to the active device.
    demo_model.to(DEFAULT_DEVICE)

    # 4. Check that the optimizer still references the model parameters.
    demo_model_p = next(demo_model.parameters())
    demo_opt_p = demo_optimizer.param_groups[0]["params"][0]

    print(f"Model parameter device: {demo_model_p.device}")
    print(f"Optimizer tracked parameter device: {demo_opt_p.device}")
    print(f"Are they the same object? {id(demo_model_p) == id(demo_opt_p)}")

    # 5. Attempt a dummy training step on the same device as the model.
    try:
        step_device = next(demo_model.parameters()).device
        input_data = torch.randn(5, 10, device=step_device)
        demo_output = demo_model(input_data)
        demo_loss = demo_output.sum()
        demo_loss.backward()
        demo_optimizer.step()
        print("Step completed successfully.")
    except Exception as e:
        print(f"Caught error: {e}")
    return


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
    # `grad` is imported in the setup cell and used below.
    grad
    return


@app.function
## Define the function
def f(x, y):
    return torch.sin(x**2 * y)


@app.cell
def _(x, y, z):
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
    # Set seed for reproducibility so the module comparison below is stable.
    torch.manual_seed(42)


    class DenseLayer(nn.Module):
        """Reference dense block implemented with high-level PyTorch modules.

        This version mirrors `nn.Linear(...); nn.ReLU()` directly. We keep it as a
        readable baseline before reimplementing the same behavior using explicit
        parameters in `DenseLayer2`.
        """

        def __init__(self, n_features: int):
            super().__init__()
            self.linlayer = nn.Sequential(nn.Linear(n_features, 1), nn.ReLU())

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return self.linlayer(X)

    return (DenseLayer,)


@app.cell
def _(DenseLayer):
    # Consolidated into the custom Dense module cell.
    Dense_1 = DenseLayer(10).to(DEFAULT_DEVICE)

    # Run one forward/backward pass so we can inspect gradients and compare against
    # the parameter-based implementation below.
    dummy_data = torch.randn(5, 10, device=next(Dense_1.parameters()).device)
    dense_1_out = Dense_1(dummy_data)
    dense_1_out.sum().backward()

    print("DenseLayer output shape:", dense_1_out.shape)
    for name, param in Dense_1.named_parameters():
        print(f"Parameter: {name}, gradient shape: {None if param.grad is None else tuple(param.grad.shape)}")
    return (Dense_1,)


@app.cell
def _(Dense_1):
    class DenseLayer2(nn.Module):
        """Dense block implemented with explicit parameters plus functional ops.

        The layer owns `weight` and `bias` directly via `nn.Parameter`, then uses
        `F.linear` followed by `torch.relu` in `forward`. This is the lower-level
        equivalent of `DenseLayer` and is useful for understanding how `nn.Linear`
        is built.
        """

        def __init__(self, in_features: int, out_features: int = 1):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.empty(out_features))

            # Match PyTorch's standard linear-layer initialization.
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            fan_in = self.weight.size(1)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return torch.relu(F.linear(X, self.weight, self.bias))


    Dense_2 = DenseLayer2(Dense_1.linlayer[0].in_features).to(
        next(Dense_1.parameters()).device
    )
    return (DenseLayer2,)


app._unparsable_cell(
    r"""
    # Copy DenseLayer's learned parameters so the two implementations can be
    # compared directly on the same input batch.
    with torch.no_grad():
        Dense_2.weight.copy_(Dense_1.linlayer[0].weight)
        Dense_2.bias.copy_(Dense_1.linlayer[0].bias)

    dense_2_out = Dense_2(dummy_data)
    print("Outputs match:", torch.allclose(dense_1_out.detach(), dense_2_out.detach()))
    print("DenseLayer2 output shape:", dense_2_out.shape)the parameter-based Dense module cell.
    """,
    name="_"
)


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ### 15. Build and train a classification MLP on the CoverType dataset
    Step 1: Load the dataset using sklearn.datasets.fetch_covtype() and create a custom PyTorch Dataset for this data.
    """)
    return


@app.cell
def _():
    data = fetch_covtype()

    # Standardize the raw features once at load time so every experiment sees the
    # same normalized representation.
    scaler = StandardScaler()
    data.data = scaler.fit_transform(data.data)


    class CoverTypeDataset(Dataset[tuple[Tensor, Tensor]]):
        """Wrap the CoverType arrays as a PyTorch dataset.

        The target labels are shifted from 1..7 to 0..6 so they line up with
        `nn.CrossEntropyLoss`, which expects zero-based class indices.
        """

        def __init__(self, data):
            self.X = torch.tensor(data.data, dtype=torch.float32)
            self.y = torch.tensor(data.target - 1, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
            return self.X[index], self.y[index]


    dataset = CoverTypeDataset(data)
    return (dataset,)


@app.cell
def _():
    dataset_summary = globals().get("dataset")
    if dataset_summary is None:
        print("Dataset cell has not finished running yet.")
    else:
        print(f"Dataset size: {len(dataset_summary)}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Step 2: Create data loaders for training, validation and testing
    """)
    return


@app.cell
def _(dataset):
    # Split the dataset once so every experiment uses the same partition.
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.25, random_state=42
    )

    # Use TensorDataset because the fully materialized tensors are already in
    # memory. This keeps the data path simple and avoids notebook-specific worker
    # pickling issues.
    base_dataset = TensorDataset(dataset.X, dataset.y)
    DEFAULT_BATCH_SIZE = 512


    def build_dataloaders(batch_size: int = DEFAULT_BATCH_SIZE):
        """Return train/validation/test loaders for the CoverType tensors.

        Parameters
        ----------
        batch_size:
            Number of examples per batch.

        Notes
        -----
        `num_workers` is fixed at zero in this notebook because the data already
        lives in memory as tensors. Adding worker processes here increased overhead
        and did not improve throughput in practice.
        """
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        }

        train_loader = DataLoader(
            Subset(base_dataset, train_indices), shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            Subset(base_dataset, val_indices), shuffle=False, **loader_kwargs
        )
        test_loader = DataLoader(
            Subset(base_dataset, test_indices), shuffle=False, **loader_kwargs
        )
        return train_loader, val_loader, test_loader


    train_loader, val_loader, test_loader = build_dataloaders()

    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)} | batch_size={DEFAULT_BATCH_SIZE}"
    )
    return build_dataloaders, train_loader, val_loader


@app.cell(hide_code=True)
def _():
    mo.md("""
    Step 3: Build a custom MLP module to tacke this classification task. You can optionally use the custome `Dense` module from the previous exercise
    """)
    return


@app.cell
def _(DenseLayer2):
    class CovTypeModel(nn.Module):
        """MLP classifier for the CoverType dataset.

        Parameters
        ----------
        input_dim:
            Number of input features.
        hidden_dims:
            Sequence of hidden-layer widths. Each hidden layer uses `DenseLayer2`.
        output_dim:
            Number of classes.
        """

        def __init__(self, input_dim: int, hidden_dims, output_dim: int):
            super().__init__()
            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.append(DenseLayer2(prev_dim, h_dim))
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    return (CovTypeModel,)


@app.cell
def _(CovTypeModel):
    # Build the baseline model for the CoverType classifier.
    #
    # The final architecture used for longer runs can be replaced later by the
    # Optuna-selected configuration, but this baseline keeps the rest of the
    # notebook runnable and gives us a stable object for smoke tests.
    model = CovTypeModel(
        input_dim=54,
        hidden_dims=[128, 256, 64],
        output_dim=7,
    ).to(DEFAULT_DEVICE)

    model
    return (model,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    Step 4: Train the model on the GPU and tro to reach 93% accuracy on the test set. For this, you will likely have to perform hyperparameter search to find the right number of layers and neurons per layer, a good learning rate and batch size and so on, optionally using Optuna for this.
    """)
    return


@app.cell
def _():
    # Compact training helpers for notebook use.
    #
    # These helpers are shared by ad hoc notebook experiments and the Optuna search.
    # They keep the default output minimal while still exposing more detailed
    # performance metrics when `verbose=True`.
    DEVICE = DEFAULT_DEVICE


    def move_batch_to_device(
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        device: torch.device,
    ):
        """Move one mini-batch to the target device.

        We request non-blocking transfers on CUDA so pinned-memory loaders can take
        advantage of asynchronous host-to-device copies.
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
        criterion=None,
        *,
        device: torch.device = DEVICE,
        max_batches: int | None = None,
    ):
        """Evaluate a model on validation or test data.

        Parameters
        ----------
        model:
            Model to evaluate.
        data_loader:
            Validation or test loader.
        criterion:
            Optional loss function. When omitted, the function reports accuracy only.
        device:
            Device on which evaluation should run.
        max_batches:
            Optional cap used for smoke tests.
        """
        model.eval()
        total_correct = torch.zeros((), device=device)
        total_examples = 0
        total_loss = torch.zeros((), device=device) if criterion is not None else None

        for batch_index, (X_batch, y_batch) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            X_batch, y_batch = move_batch_to_device(X_batch, y_batch, device)
            logits = model(X_batch)
            batch_size = y_batch.size(0)
            if total_loss is not None:
                total_loss += criterion(logits, y_batch).detach() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum()
            total_examples += batch_size

        metrics = {
            "accuracy": (total_correct / total_examples).item(),
            "examples": total_examples,
        }
        if total_loss is not None:
            metrics["loss"] = (total_loss / total_examples).item()
        return metrics


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
        show_epoch_summary: bool = True,
        verbose: bool = False,
        trial=None,
        hparams: dict | None = None,
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
        log_interval: int | None = None,
        profile_training: bool = False,
        profile_dir: str | None = None,
        stop_event=None,
        progress_callback=None,
    ):
        """Train a model and return the run history plus TensorBoard log path.

        Parameters
        ----------
        model, optimizer, criterion:
            Standard PyTorch training objects.
        train_loader, val_loader:
            Data loaders for training and validation.
        epochs:
            Number of passes over the training data.
        device:
            Device on which training should run.
        run_dir, run_name:
            TensorBoard output location.
        show_epoch_summary:
            Print the basic epoch metrics every epoch.
        verbose:
            Print extra performance diagnostics such as throughput and CUDA memory.
        trial:
            Optional Optuna trial for pruning/reporting.
        hparams:
            Optional hyperparameters to record in TensorBoard.
        max_train_batches, max_eval_batches:
            Optional caps for smoke tests.
        log_interval:
            Optional batch interval for progress prints inside long epochs.
        profile_training:
            Enable a short profiler trace during the first epoch.
        profile_dir:
            Output directory for profiler traces.
        stop_event:
            Optional threading.Event that can be set to request early stopping after the current epoch.
        progress_callback:
            Optional callback used by notebook progress bar UIs.

        Returns
        -----------
        history:
            Dictionary containing training/validation metrics and performance stats.
        log_dir:
            Path where TensorBoard logs were written.
        """
        run_root = Path(run_dir)
        run_root.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_dir = str(run_root / run_name)
        writer = SummaryWriter(log_dir=log_dir)

        if profile_dir is None:
            profile_dir = str(run_root / "profiler" / run_name)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "epoch_time": [],
            "samples_per_sec": [],
            "avg_step_time_ms": [],
            "avg_data_time_ms": [],
            "best_val_accuracy": 0.0,
            "best_epoch": 0,
            "stopped_early": False,
        }

        for epoch in range(epochs):
            model.train()
            epoch_start = time.perf_counter()
            batch_fetch_start = epoch_start
            data_time_total = 0.0
            step_time_total = 0.0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            running_loss = torch.zeros((), device=device)
            running_correct = torch.zeros((), device=device)
            seen_examples = 0
            step_count = 0
            profiler = None
            train_batches_total = (
                min(len(train_loader), max_train_batches)
                if max_train_batches is not None
                else len(train_loader)
            )

            if profile_training and epoch == 0:
                activities = [ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                profiler = profile(
                    activities=activities,
                    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                profiler.__enter__()

            try:
                for batch_index, (X_batch, y_batch) in enumerate(train_loader):
                    if max_train_batches is not None and batch_index >= max_train_batches:
                        break

                    data_ready = time.perf_counter()
                    data_time_total += data_ready - batch_fetch_start
                    step_start = time.perf_counter()

                    # Separate the transfer and training-step regions so profiler
                    # traces show where time is actually going.
                    with (
                        record_function("train/data_to_device")
                        if profiler is not None
                        else contextlib.nullcontext()
                    ):
                        X_batch, y_batch = move_batch_to_device(X_batch, y_batch, device)

                    with (
                        record_function("train/step")
                        if profiler is not None
                        else contextlib.nullcontext()
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                        loss.backward()
                        optimizer.step()

                    if profiler is not None:
                        profiler.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                    step_time_total += time.perf_counter() - step_start
                    batch_size = y_batch.size(0)
                    running_loss += loss.detach() * batch_size
                    running_correct += (logits.argmax(dim=1) == y_batch).sum()
                    seen_examples += batch_size
                    step_count += 1
                    batch_fetch_start = time.perf_counter()

                    if progress_callback is not None:
                        progress_callback(
                            increment=1,
                            subtitle=f"Epoch {epoch + 1}/{epochs} | batch {batch_index + 1}/{train_batches_total}",
                        )

                    if verbose and log_interval is not None and (batch_index + 1) % log_interval == 0:
                        print(f"    Batch {batch_index + 1}: loss={loss.item():.4f}")
            finally:
                if profiler is not None:
                    profiler.__exit__(None, None, None)

            if seen_examples == 0:
                break

            train_loss = (running_loss / seen_examples).item()
            train_accuracy = (running_correct / seen_examples).item()
            val_metrics = evaluate_model(model, val_loader, device=device, max_batches=max_eval_batches)
            epoch_time = time.perf_counter() - epoch_start
            samples_per_sec = seen_examples / step_time_total if step_time_total else 0.0
            avg_step_time_ms = 1000.0 * step_time_total / step_count if step_count else 0.0
            avg_data_time_ms = 1000.0 * data_time_total / step_count if step_count else 0.0

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["epoch_time"].append(epoch_time)
            history["samples_per_sec"].append(samples_per_sec)
            history["avg_step_time_ms"].append(avg_step_time_ms)
            history["avg_data_time_ms"].append(avg_data_time_ms)

            if val_metrics["accuracy"] > history["best_val_accuracy"]:
                history["best_val_accuracy"] = val_metrics["accuracy"]
                history["best_epoch"] = epoch + 1

            writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
            writer.add_scalar("epoch/train_accuracy", train_accuracy, epoch + 1)
            writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch + 1)
            writer.add_scalar("epoch/time_sec", epoch_time, epoch + 1)
            writer.add_scalar("perf/samples_per_sec", samples_per_sec, epoch + 1)
            writer.add_scalar("perf/avg_step_time_ms", avg_step_time_ms, epoch + 1)
            writer.add_scalar("perf/avg_data_time_ms", avg_data_time_ms, epoch + 1)

            peak_allocated_mb = None
            peak_reserved_mb = None
            if device.type == "cuda":
                peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
                writer.add_scalar("perf/max_cuda_mem_allocated_mb", peak_allocated_mb, epoch + 1)
                writer.add_scalar("perf/max_cuda_mem_reserved_mb", peak_reserved_mb, epoch + 1)

            if show_epoch_summary:
                print(
                    f"Epoch {epoch + 1}/{epochs} | time={epoch_time:.2f}s | "
                    f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )
            if verbose:
                print(
                    f"  Throughput: {samples_per_sec:.1f} samples/s | "
                    f"avg_step={avg_step_time_ms:.2f} ms | avg_data={avg_data_time_ms:.2f} ms"
                )
                if peak_allocated_mb is not None and peak_reserved_mb is not None:
                    print(
                        f"  CUDA peak memory: allocated={peak_allocated_mb:.1f} MB | "
                        f"reserved={peak_reserved_mb:.1f} MB"
                    )
                if profile_training and epoch == 0:
                    print(f"  Profiler trace written to: {profile_dir}")

            if trial is not None:
                trial.report(val_metrics["accuracy"], step=epoch)
                if trial.should_prune():
                    writer.close()
                    raise optuna.TrialPruned()

            if progress_callback is not None:
                progress_callback(
                    increment=0,
                    subtitle=f"Epoch {epoch + 1}/{epochs} complete | val_acc={val_metrics['accuracy']:.4f}",
                )

            if stop_event is not None and stop_event.is_set():
                history["stopped_early"] = True
                break

        if hparams and history["val_accuracy"]:
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)

    print(f"Training device: {DEVICE}")
    return criterion, optimizer


@app.cell
def _(CovTypeModel, DEVICE, build_dataloaders, criterion, train_model):
    ## Optional: Optuna search helpers.

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
        """Create an Optuna objective bound to the current notebook helpers.

        The returned `objective(trial)` closure matches Optuna's required API while
        still letting the notebook pass through fixed configuration such as epoch
        count or smoke-test batch caps.
        """

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
            hidden_dims_key = trial.suggest_categorical("hidden_dims_key", list(HIDDEN_DIM_OPTIONS.keys()))
            hidden_dims = HIDDEN_DIM_OPTIONS[hidden_dims_key]

            trial_model = CovTypeModel(input_dim=54, hidden_dims=hidden_dims, output_dim=7).to(DEVICE)
            trial_optimizer = optim.SGD(trial_model.parameters(), lr=lr, momentum=0.9)
            trial_train_loader, trial_val_loader, _ = build_dataloaders(batch_size=batch_size)

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
                show_epoch_summary=False,
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
        The study is persisted under a new default name because Optuna stores the
        search-space schema. Older studies used an incompatible architecture
        encoding, which is what caused the earlier categorical `ValueError`.
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

        The search is intentionally triggered manually rather than reactively. That
        keeps notebook edits from accidentally launching a long study.
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
    return


@app.cell(column=3)
def _():
    # Optuna is manual on purpose.
    #
    # Example:
    # run_optuna_search(n_trials=10, epochs=5)
    return


@app.cell(hide_code=True)
def _():
    cancelled = Event()
    training_result = {}

    mo.md(
        "Shared training state: `cancelled` is the stop event and "
        "`training_result` stores the most recent `history` and `log_dir`."
    )
    return cancelled, training_result


@app.cell
def _(
    DEVICE,
    cancelled,
    criterion,
    model,
    optimizer,
    train_loader,
    train_model,
    training_result,
    val_loader,
):
    # Best parameters found in the
    BEST_PARAMS = {
        "lr": 0.0010253509690168502,
        "batch_size": 128,
        "hidden_dims_key": "128-256-64",
    }


    epochs = 4

    def training_loop():
        with mo.status.progress_bar(
            total=epochs * len(train_loader),
            # title="Manual training",
            # subtitle="Preparing training loop",
        ) as pbar:
            history, log_dir = train_model(
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                epochs=epochs,
                device=DEVICE,
                run_dir="./runs/test_3",
                verbose=True,
                stop_event=cancelled,
                progress_callback=lambda **kwargs: pbar.update(**kwargs),
            )

        if history["stopped_early"]:
            print("Training stopped after the current epoch by request.")
        else:
            print("Training completed the requested epochs.")
        print(f"TensorBoard logs: {log_dir}")

        training_result["history"] = history
        training_result["log_dir"] = log_dir


    cancelled.clear()
    training_result.clear()
    mo.Thread(target=training_loop).start()
    mo.md(
        "Training started in the background. Use the stop button in the cell above to stop after the current epoch."
    )
    return


@app.cell
def _(cancelled):
    cancel = mo.ui.button(
        label="Stop after current epoch",
        kind="warn",
        on_change=lambda _: cancelled.set(),
    )

    mo.vstack(
        [
            mo.md(
                "Click **Stop after current epoch** while the training thread is running. "
                "Rerunning the training cell automatically clears the previous stop request."
            ),
            cancel,
        ]
    )
    return


@app.cell
def _(training_result):
    training_result
    return


if __name__ == "__main__":
    app.run()
