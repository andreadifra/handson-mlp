import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import contextlib
    import marimo as mo
    import optuna
    import time
    import json

    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    from typing import Any, Callable

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


@app.cell
def _():
    # Shared device used throughout the notebook. Cells should use this instead of
    # hard-coding CUDA strings so the notebook still runs on CPU-only machines.
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (DEFAULT_DEVICE,)


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
def _(DEFAULT_DEVICE):
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
def _(DEFAULT_DEVICE):
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
def _(DEFAULT_DEVICE, DenseLayer):
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
    return Dense_1, dense_1_out, dummy_data


@app.class_definition
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


@app.cell
def _(Dense_1, dense_1_out, dummy_data):
    # Copy DenseLayer's learned parameters so the two implementations can be
    # compared directly on the same input batch.
    Dense_2 = DenseLayer2(Dense_1.linlayer[0].in_features).to(
        next(Dense_1.parameters()).device
    )
    with torch.no_grad():
        Dense_2.weight.copy_(Dense_1.linlayer[0].weight)
        Dense_2.bias.copy_(Dense_1.linlayer[0].bias)

    dense_2_out = Dense_2(dummy_data)
    print(
        "Outputs match:",
        torch.allclose(dense_1_out.detach(), dense_2_out.detach()),
    )
    print("DenseLayer2 output shape:", dense_2_out.shape)
    return


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
    DEFAULT_TRAIN_BATCH_SIZE = 512
    DEFAULT_EVAL_BATCH_SIZE_MULTIPLIER = 4


    def _auto_eval_batch_size(train_batch_size: int, split_size: int) -> int:
        """Pick a larger evaluation batch size without exceeding the split size."""
        return max(train_batch_size, min(split_size, train_batch_size * DEFAULT_EVAL_BATCH_SIZE_MULTIPLIER))


    def build_dataloaders(
        train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
        eval_batch_size: int | None = None,
        test_batch_size: int | None = None,
        *,
        source_dataset=base_dataset,
        train_idx=train_indices,
        val_idx=val_indices,
        test_idx=test_indices,
    ):
        """Return train/validation/test loaders for the CoverType tensors.

        Parameters
        ----------
        train_batch_size:
            Mini-batch size used for gradient updates.
        eval_batch_size, test_batch_size:
            Batch sizes used for validation and test passes. When omitted, these
            default to a larger multiple of the training batch size because they do
            not affect optimization dynamics.
        source_dataset, train_idx, val_idx, test_idx:
            Explicit dataset and split inputs. Defaults keep notebook calls compact
            while avoiding hidden closure-only behavior.

        Notes
        -----
        `num_workers` is fixed at zero in this notebook because the data already
        lives in memory as tensors. Adding worker processes here increased overhead
        and did not improve throughput in practice.
        """
        eval_batch_size = eval_batch_size or _auto_eval_batch_size(train_batch_size, len(val_idx))
        test_batch_size = test_batch_size or _auto_eval_batch_size(train_batch_size, len(test_idx))
        base_loader_kwargs = {
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        }

        train_loader = DataLoader(
            Subset(source_dataset, train_idx),
            batch_size=train_batch_size,
            shuffle=True,
            **base_loader_kwargs,
        )
        val_loader = DataLoader(
            Subset(source_dataset, val_idx),
            batch_size=eval_batch_size,
            shuffle=False,
            **base_loader_kwargs,
        )
        test_loader = DataLoader(
            Subset(source_dataset, test_idx),
            batch_size=test_batch_size,
            shuffle=False,
            **base_loader_kwargs,
        )
        return train_loader, val_loader, test_loader


    train_loader, val_loader, test_loader = build_dataloaders()

    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)} | train_batch_size={DEFAULT_TRAIN_BATCH_SIZE}, "
        f"eval_batch_size={val_loader.batch_size}, test_batch_size={test_loader.batch_size}"
    )
    return (build_dataloaders,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Step 3: Build a custom MLP module to tacke this classification task. You can optionally use the custome `Dense` module from the previous exercise
    """)
    return


@app.class_definition
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


@app.cell
def _(DEFAULT_DEVICE):
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
def _(DEFAULT_DEVICE):
    # Compact training helpers for notebook use.
    #
    # These helpers are shared by ad hoc notebook experiments and the Optuna search.
    # They keep the default output minimal while still exposing more detailed
    # performance metrics when `verbose=True`.
    DEVICE = DEFAULT_DEVICE
    return (DEVICE,)


@app.function
def move_batch_to_device(X_batch: torch.Tensor, y_batch: torch.Tensor, device: torch.device):
    """Move one mini-batch to the target device.

    Parameters
    ----------
    X_batch:
        Batch of input tensors as returned by a PyTorch ``DataLoader``.
    y_batch:
        Batch of target tensors paired with ``X_batch``.
    device:
        Target ``torch.device``. CUDA transfers request ``non_blocking=True`` so
        DataLoaders with pinned memory can overlap host-to-device copies with GPU
        work. CPU runs use ordinary blocking transfers.

    Returns
    -------
    X_batch, y_batch:
        The same tensors moved to ``device``.
    """
    if device.type == "cuda":
        return (
            X_batch.to(device, non_blocking=True),
            y_batch.to(device, non_blocking=True),
        )
    return X_batch.to(device), y_batch.to(device)


@app.function
def evaluate_classification_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module | None = None,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Evaluate a classification model on a data loader.

    This is the default evaluator for the CoverType exercise. It owns the metric
    contract used by ``train_model``: it names the primary metric, reports the
    primary value, declares whether higher values are better, and includes all
    scalar validation metrics in ``metrics``.

    Parameters
    ----------
    model:
        PyTorch classifier that returns class logits with shape
        ``(batch_size, n_classes)``.
    data_loader:
        Validation or test loader yielding ``(X_batch, y_batch)`` pairs.
    criterion:
        Optional loss function. When provided, the returned metrics include
        average ``loss`` weighted by batch size.
    device:
        Device on which evaluation should run.
    max_batches:
        Optional cap on evaluation batches, mainly for smoke tests and quick
        Optuna trials.

    Returns
    -------
    dict[str, object]
        Evaluation result with ``primary_metric="accuracy"``, ``primary_value``
        set to validation accuracy, ``higher_is_better=True``, and a ``metrics``
        dictionary containing ``accuracy``, ``examples``, and optionally ``loss``.
    """
    model.eval()
    total_correct = torch.zeros((), device=device)
    total_examples = 0
    total_loss = torch.zeros((), device=device) if criterion is not None else None

    with torch.inference_mode():
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

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty data loader or zero selected batches.")

    accuracy = (total_correct / total_examples).item()
    metrics = {
        "accuracy": accuracy,
        "examples": float(total_examples),
    }
    if total_loss is not None:
        metrics["loss"] = (total_loss / total_examples).item()

    return {
        "primary_metric": "accuracy",
        "primary_value": accuracy,
        "higher_is_better": True,
        "metrics": metrics,
    }


@app.function
def make_training_history(history: dict | None = None) -> dict:
    """Return a metric-agnostic history dictionary for ``train_model``.

    The evaluator, not ``train_model``, chooses the primary validation metric and
    its direction. History records those choices after the first validation pass
    so a continued run can verify that the evaluator contract has not changed.

    Parameters
    ----------
    history:
        Existing history to continue, usually from an earlier call to
        ``train_model`` in the same notebook session. Older accuracy-specific
        histories are migrated when possible.

    Returns
    -------
    dict
        Mutable history dictionary with lists for epoch metrics, best-metric
        metadata, checkpoint path metadata, and early-stop status.
    """
    default_history = {
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],
        "val_metrics": {},
        "epoch_time": [],
        "samples_per_sec": [],
        "avg_step_time_ms": [],
        "avg_data_time_ms": [],
        "best_val_metric": None,
        "best_epoch": 0,
        "primary_metric": None,
        "higher_is_better": None,
        "stopped_early": False,
        "checkpoint_path": None,
    }
    if history is None:
        return default_history

    # Migrate older accuracy-specific histories so long-running notebook state can
    # continue with the generic metric names.
    if "val_metric" not in history and "val_accuracy" in history:
        history["val_metric"] = list(history["val_accuracy"])
        history.setdefault("primary_metric", "accuracy")
        history.setdefault("higher_is_better", True)

    if "best_val_metric" not in history and "best_val_accuracy" in history:
        history["best_val_metric"] = history.get("best_val_accuracy")

    for key, value in default_history.items():
        history.setdefault(key, value)

    history["stopped_early"] = False
    return history


@app.function
def save_training_checkpoint(
    checkpoint_path,
    *,
    model,
    optimizer,
    history: dict,
    epoch: int,
    metadata: dict | None = None,
):
    """Save the state needed to resume a model trained by ``train_model``.

    The checkpoint stores the generic best validation metric plus the evaluator's
    metric name/direction so the file remains meaningful for classification and
    regression runs.

    Parameters
    ----------
    checkpoint_path:
        Destination path for the checkpoint file. Parent directories are created
        automatically.
    model:
        PyTorch model whose ``state_dict`` should be saved.
    optimizer:
        Optimizer whose ``state_dict`` should be saved so training can continue
        with the same momentum/adaptive state.
    history:
        Training history returned by ``train_model``. The checkpoint records this
        verbatim, including ``best_val_metric`` and the evaluator-selected
        ``primary_metric``.
    epoch:
        One-based epoch number corresponding to the saved model state.
    metadata:
        Optional extra context, such as Optuna study name, trial number, or
        hyperparameters.

    Returns
    -------
    checkpoint_path:
        The path that was written, returned for convenient logging.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_metric": history["best_val_metric"],
            "best_epoch": history["best_epoch"],
            "primary_metric": history["primary_metric"],
            "higher_is_better": history["higher_is_better"],
            "metadata": metadata or {},
        },
        checkpoint_path,
    )
    return checkpoint_path


@app.function
def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    *,
    epochs: int = 10,
    device: torch.device,
    evaluate_fn,
    run_dir: str = "./runs",
    run_name: str | None = None,
    start_epoch: int = 0,
    history: dict | None = None,
    checkpoint_dir=None,
    save_best: bool = False,
    checkpoint_metadata: dict | None = None,
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
    """Train a PyTorch model and return metric-agnostic run history.

    ``train_model`` owns mechanics common across supervised tasks: gradient
    updates, validation calls, TensorBoard logging, optional profiling, Optuna
    pruning, cancellation, and best-checkpoint saving. Task-specific validation
    is delegated to ``evaluate_fn``, which owns the primary metric name and
    comparison direction.

    Parameters
    ----------
    model:
        PyTorch model to train. It should return predictions compatible with
        ``criterion`` for each ``X_batch`` from ``train_loader``.
    optimizer:
        Optimizer already constructed for ``model.parameters()``.
    criterion:
        Loss function used for the training step. It is also passed to
        ``evaluate_fn`` so validation loss can be reported when appropriate.
    train_loader, val_loader:
        DataLoaders yielding ``(X_batch, y_batch)`` pairs for training and
        validation.
    epochs:
        Number of additional epochs to run from ``start_epoch``.
    device:
        Target ``torch.device`` for model inputs, targets, losses, and metrics.
    evaluate_fn:
        Validation function with the signature
        ``evaluate_fn(model, val_loader, criterion, device=device, max_batches=...)``.
        It must return a dictionary with ``primary_metric`` (name),
        ``primary_value`` (float-like), ``higher_is_better`` (bool), and
        ``metrics`` (a dictionary of scalar validation metrics). This explicit
        evaluator keeps classification and regression choices out of the generic
        training loop.
    run_dir, run_name:
        TensorBoard output location. Reusing the same ``run_name`` appends event
        files to the same TensorBoard run.
    start_epoch:
        Epoch offset used for TensorBoard global steps and printed epoch numbers
        when continuing a live run.
    history:
        Existing history to append to when continuing a run. Its stored primary
        metric name and direction must match the current evaluator result.
    checkpoint_dir:
        Directory for the best-model checkpoint. If omitted and ``save_best`` is
        true, checkpoints are written under ``log_dir/checkpoints``.
    save_best:
        Save ``best.pt`` whenever the evaluator's primary metric improves. No
        checkpoint is written unless this is true.
    checkpoint_metadata:
        Optional metadata stored inside each checkpoint, for example Optuna study
        name, trial number, or hyperparameters.
    show_epoch_summary:
        Print compact epoch metrics after validation.
    verbose:
        Print extra diagnostics such as throughput, CUDA memory, and optional
        profiler location.
    trial:
        Optional Optuna trial. When provided, the validation metric is reported
        each epoch and pruning is honored.
    hparams:
        Optional hyperparameters to record with TensorBoard's HParams plugin.
        Values that are not bool/int/float/str are stringified for compatibility.
    max_train_batches, max_eval_batches:
        Optional batch caps for smoke tests or deliberately short Optuna trials.
    log_interval:
        Optional training-batch interval for loss prints when ``verbose`` is true.
    profile_training:
        Enable a short PyTorch profiler trace during the first epoch.
    profile_dir:
        Output directory for profiler traces. Defaults under ``run_dir``.
    stop_event:
        Optional ``threading.Event``. If set after an epoch, training stops cleanly
        and ``history["stopped_early"]`` is marked true.
    progress_callback:
        Optional callback for notebook progress UIs. It is called with keyword
        arguments such as ``increment`` and ``subtitle``.

    Returns
    -------
    history:
        Dictionary containing training loss, validation metrics, performance
        stats, best-metric metadata, checkpoint path, and early-stop status.
    log_dir:
        Path where TensorBoard logs were written.
    """
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative.")

    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = str(run_root / run_name)
    writer = SummaryWriter(log_dir=log_dir)

    if profile_dir is None:
        profile_dir = str(run_root / "profiler" / run_name)

    history = make_training_history(history)
    checkpoint_path = None
    if save_best:
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else Path(log_dir) / "checkpoints"
        checkpoint_path = checkpoint_root / "best.pt"

    try:
        for epoch in range(epochs):
            global_epoch = start_epoch + epoch + 1
            model.train()
            epoch_start = time.perf_counter()
            batch_fetch_start = epoch_start
            data_time_total = 0.0
            step_time_total = 0.0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            running_loss = torch.zeros((), device=device)
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

                    # Keep transfer and optimization separate in profiler traces.
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
                        predictions = model(X_batch)
                        loss = criterion(predictions, y_batch)
                        loss.backward()
                        optimizer.step()

                    if profiler is not None:
                        profiler.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                    step_time_total += time.perf_counter() - step_start
                    batch_size = y_batch.size(0)
                    running_loss += loss.detach() * batch_size
                    seen_examples += batch_size
                    step_count += 1
                    batch_fetch_start = time.perf_counter()

                    if progress_callback is not None:
                        progress_callback(
                            increment=1,
                            subtitle=(
                                f"Epoch {global_epoch} | "
                                f"batch {batch_index + 1}/{train_batches_total}"
                            ),
                        )

                    if verbose and log_interval is not None and (batch_index + 1) % log_interval == 0:
                        print(f"    Batch {batch_index + 1}: loss={loss.item():.4f}")
            finally:
                if profiler is not None:
                    profiler.__exit__(None, None, None)

            if seen_examples == 0:
                break

            train_loss = (running_loss / seen_examples).item()
            evaluation = evaluate_fn(
                model,
                val_loader,
                criterion,
                device=device,
                max_batches=max_eval_batches,
            )
            if not isinstance(evaluation, dict):
                raise TypeError("evaluate_fn must return a dictionary.")

            missing_keys = {
                "primary_metric",
                "primary_value",
                "higher_is_better",
                "metrics",
            } - set(evaluation)
            if missing_keys:
                missing_text = ", ".join(sorted(missing_keys))
                raise KeyError(f"evaluate_fn result is missing required keys: {missing_text}.")

            primary_metric = str(evaluation["primary_metric"])
            if not primary_metric:
                raise ValueError("evaluate_fn returned an empty primary_metric name.")
            val_metric = float(evaluation["primary_value"])
            higher_is_better = evaluation["higher_is_better"]
            if not isinstance(higher_is_better, bool):
                raise TypeError("evaluate_fn result key 'higher_is_better' must be a bool.")

            val_metrics = evaluation["metrics"]
            if not isinstance(val_metrics, dict):
                raise TypeError("evaluate_fn result key 'metrics' must be a dictionary.")
            val_metrics = dict(val_metrics)
            val_metrics[primary_metric] = val_metric
            if "loss" in evaluation and "loss" not in val_metrics:
                val_metrics["loss"] = evaluation["loss"]

            existing_metric = history.get("primary_metric")
            if existing_metric is None:
                history["primary_metric"] = primary_metric
            elif existing_metric != primary_metric:
                raise ValueError(
                    "Cannot continue a history tracked with "
                    f"primary_metric={existing_metric!r} using evaluator metric {primary_metric!r}."
                )

            existing_direction = history.get("higher_is_better")
            if existing_direction is None:
                history["higher_is_better"] = higher_is_better
            elif existing_direction != higher_is_better:
                raise ValueError("Cannot continue a history with a different metric direction.")

            if history["best_val_metric"] is None and history["val_metric"]:
                best_epoch, best_metric = (
                    max(enumerate(history["val_metric"], start=1), key=lambda item: item[1])
                    if higher_is_better
                    else min(enumerate(history["val_metric"], start=1), key=lambda item: item[1])
                )
                history["best_epoch"] = best_epoch
                history["best_val_metric"] = best_metric

            val_loss = val_metrics.get("loss")
            val_loss = float(val_loss) if val_loss is not None else None
            metric_tag = primary_metric.strip().lower().replace(" ", "_").replace("/", "_")
            epoch_time = time.perf_counter() - epoch_start
            samples_per_sec = seen_examples / step_time_total if step_time_total else 0.0
            avg_step_time_ms = 1000.0 * step_time_total / step_count if step_count else 0.0
            avg_data_time_ms = 1000.0 * data_time_total / step_count if step_count else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)
            history["epoch_time"].append(epoch_time)
            history["samples_per_sec"].append(samples_per_sec)
            history["avg_step_time_ms"].append(avg_step_time_ms)
            history["avg_data_time_ms"].append(avg_data_time_ms)

            # Keep the primary metric generic, but retain all scalar validation
            # metrics by their evaluator-provided names for later analysis.
            metric_history = history.setdefault("val_metrics", {})
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    metric_history.setdefault(metric_name, []).append(float(metric_value))

            previous_best = history["best_val_metric"]
            improved = (
                previous_best is None
                or (higher_is_better and val_metric > previous_best)
                or (not higher_is_better and val_metric < previous_best)
            )
            if improved:
                history["best_val_metric"] = val_metric
                history["best_epoch"] = global_epoch

                if checkpoint_path is not None:
                    checkpoint_start = time.perf_counter()
                    saved_path = save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        history=history,
                        epoch=global_epoch,
                        metadata=checkpoint_metadata,
                    )
                    checkpoint_time = time.perf_counter() - checkpoint_start
                    history["checkpoint_path"] = str(saved_path)

                    previous_text = "none" if previous_best is None else f"{previous_best:.4f}"
                    checkpoint_message = (
                        f"Saved best checkpoint at epoch {global_epoch}: "
                        f"val_{primary_metric} improved {previous_text} -> {val_metric:.4f} "
                        f"({checkpoint_time:.2f}s)"
                    )
                    print(checkpoint_message)
                    if progress_callback is not None:
                        progress_callback(increment=0, subtitle=checkpoint_message)

            writer.add_scalar("epoch/train_loss", train_loss, global_epoch)
            if val_loss is not None:
                writer.add_scalar("epoch/val_loss", val_loss, global_epoch)
            writer.add_scalar(f"epoch/val_{metric_tag}", val_metric, global_epoch)
            writer.add_scalar("epoch/time_sec", epoch_time, global_epoch)
            writer.add_scalar("perf/samples_per_sec", samples_per_sec, global_epoch)
            writer.add_scalar("perf/avg_step_time_ms", avg_step_time_ms, global_epoch)
            writer.add_scalar("perf/avg_data_time_ms", avg_data_time_ms, global_epoch)

            peak_allocated_mb = None
            peak_reserved_mb = None
            if device.type == "cuda":
                peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
                writer.add_scalar("perf/max_cuda_mem_allocated_mb", peak_allocated_mb, global_epoch)
                writer.add_scalar("perf/max_cuda_mem_reserved_mb", peak_reserved_mb, global_epoch)

            if show_epoch_summary:
                val_loss_text = "" if val_loss is None else f" | val_loss={val_loss:.4f}"
                print(
                    f"Epoch {global_epoch} | time={epoch_time:.2f}s | "
                    f"train_loss={train_loss:.4f}{val_loss_text} | "
                    f"val_{primary_metric}={val_metric:.4f}"
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
                trial.report(val_metric, step=global_epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if progress_callback is not None:
                progress_callback(
                    increment=0,
                    subtitle=f"Epoch {global_epoch} complete | val_{primary_metric}={val_metric:.4f}",
                )

            if stop_event is not None and stop_event.is_set():
                history["stopped_early"] = True
                break

        if hparams and history["val_metric"]:
            primary_metric = history["primary_metric"]
            metric_tag = primary_metric.strip().lower().replace(" ", "_").replace("/", "_")
            metric_summary = {
                f"hparam/best_val_{metric_tag}": history["best_val_metric"],
                f"hparam/final_val_{metric_tag}": history["val_metric"][-1],
                "hparam/best_epoch": history["best_epoch"],
            }
            if history["val_loss"] and history["val_loss"][-1] is not None:
                metric_summary["hparam/final_val_loss"] = history["val_loss"][-1]

            writer.add_hparams(
                {
                    key: value if isinstance(value, (bool, int, float, str)) else str(value)
                    for key, value in hparams.items()
                },
                metric_summary,
                run_name="hparams",
            )
    finally:
        writer.close()

    return history, log_dir


@app.cell
def _(DEVICE, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)

    print(f"Training device: {DEVICE}")
    return criterion, optimizer


@app.class_definition
@__import__("dataclasses").dataclass(frozen=True)
class CoverTypeSearchSpace:
    """Small, stable hyperparameter space for the CoverType exercise.

    Architecture options are keyed by strings because Optuna persists categorical
    choices. Keeping those keys stable lets existing studies be reloaded safely.

    Attributes
    ----------
    lr_low, lr_high:
        Lower and upper bounds for the log-uniform learning-rate search.
    batch_sizes:
        Candidate mini-batch sizes sampled by Optuna.
    hidden_dim_options:
        Stable categorical architecture choices. Each item pairs the persisted
        Optuna key with the hidden-layer widths used to build ``CovTypeModel``.
    """

    lr_low: float = 1e-4
    lr_high: float = 5e-2
    batch_sizes: tuple[int, ...] = (128, 256, 512, 1024)
    hidden_dim_options: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("64-128", (64, 128)),
        ("128-256", (128, 256)),
        ("128-256-64", (128, 256, 64)),
        ("256-512-128", (256, 512, 128)),
    )

    @property
    def hidden_dims_by_key(self) -> dict[str, tuple[int, ...]]:
        """Map persisted architecture keys to hidden-layer widths."""
        return dict(self.hidden_dim_options)


@app.class_definition
@__import__("dataclasses").dataclass(frozen=True)
class OptunaStudyConfig:
    """Configuration for running an Optuna study from this notebook.

    Attributes
    ----------
    study_name:
        Name used by Optuna to identify the study in persistent storage.
    direction:
        Optuna optimization direction. This must agree with the evaluator used by
        the trial: ``"maximize"`` when the evaluator reports
        ``higher_is_better=True`` and ``"minimize"`` otherwise.
    storage:
        Optional Optuna storage URL. The default SQLite file keeps study state
        inside ``notes/`` so notebook experiments can resume across sessions.
    load_if_exists:
        Reuse an existing study with the same name instead of raising an error.
    metric_name:
        Human-readable Optuna metric name shown in study summaries. This can be
        more descriptive than the evaluator's primary metric, such as
        ``validation_accuracy``.
    n_trials, timeout, n_jobs:
        Optuna search limits and parallelism settings.
    show_progress_bar:
        Whether Optuna should show its own progress bar during optimization.
    gc_after_trial:
        Ask Optuna to run garbage collection after each trial, useful in notebook
        sessions that repeatedly construct models.
    catch:
        Exception types that Optuna should catch and mark as failed trials.
    epochs:
        Number of epochs each trial trains for.
    run_dir:
        TensorBoard root directory for trial logs.
    max_train_batches, max_eval_batches:
        Optional caps for smoke tests or intentionally small trial runs.
    eval_batch_size, test_batch_size:
        Optional loader batch sizes overriding the notebook defaults.
    save_best_checkpoint:
        Whether each trial should save its best model checkpoint.
    checkpoint_dir:
        Optional root directory for trial checkpoints. Each trial gets its own
        subdirectory when checkpoint saving is enabled.
    input_dim, output_dim:
        CoverType model input feature count and class count.
    optimizer_momentum:
        Momentum value used by the SGD optimizer factory in ``run_optuna_search``.
    sampler, pruner:
        Optional Optuna sampler and pruner instances for custom search behavior.
    """

    study_name: str = "covertype_hpo_v2"
    direction: str = "maximize"
    storage: str | None = "sqlite:///notes/optuna_study.db"
    load_if_exists: bool = True
    metric_name: str = "validation_accuracy"
    n_trials: int = 10
    timeout: float | None = None
    n_jobs: int = 1
    show_progress_bar: bool = True
    gc_after_trial: bool = True
    catch: tuple[type[Exception], ...] = ()
    epochs: int = 5
    run_dir: str = "./runs/optuna"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    eval_batch_size: int | None = None
    test_batch_size: int | None = None
    save_best_checkpoint: bool = False
    checkpoint_dir: str | None = None
    input_dim: int = 54
    output_dim: int = 7
    optimizer_momentum: float = 0.9
    sampler: object | None = None
    pruner: object | None = None


@app.cell
def _():
    ## Optional: Optuna search helpers.
    DEFAULT_STUDY_NAME = "covertype_hpo_v2"
    COVERTYPE_SEARCH_SPACE = CoverTypeSearchSpace()
    HIDDEN_DIM_OPTIONS = COVERTYPE_SEARCH_SPACE.hidden_dims_by_key
    return COVERTYPE_SEARCH_SPACE, HIDDEN_DIM_OPTIONS


@app.function
def suggest_covertype_hparams(
    trial,
    search_space: CoverTypeSearchSpace = CoverTypeSearchSpace(),
) -> dict:
    """Sample one CoverType hyperparameter configuration for an Optuna trial.

    Parameters
    ----------
    trial:
        Active Optuna trial used to suggest values.
    search_space:
        Search-space definition containing learning-rate bounds, batch sizes,
        and hidden-layer options.

    Returns
    -------
    dict
        Hyperparameters for one trial. ``hidden_dims_key`` is the persisted
        categorical value; ``hidden_dims`` is the tuple used to build the model.
    """
    hidden_dims_by_key = search_space.hidden_dims_by_key
    hidden_dims_key = trial.suggest_categorical(
        "hidden_dims_key",
        list(hidden_dims_by_key),
    )
    return {
        "lr": trial.suggest_float(
            "lr",
            search_space.lr_low,
            search_space.lr_high,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            list(search_space.batch_sizes),
        ),
        "hidden_dims_key": hidden_dims_key,
        "hidden_dims": hidden_dims_by_key[hidden_dims_key],
    }


@app.function
def create_optuna_study(config: OptunaStudyConfig = OptunaStudyConfig()):
    """Create or reload an Optuna study from explicit notebook configuration.

    Parameters
    ----------
    config:
        Study configuration controlling storage, study name, direction, sampler,
        pruner, and displayed metric name.

    Returns
    -------
    optuna.Study
        Study ready to optimize. If ``config.storage`` and
        ``config.load_if_exists`` point at an existing study, that study is
        reused.
    """
    study = optuna.create_study(
        direction=config.direction,
        sampler=config.sampler,
        pruner=config.pruner,
        storage=config.storage,
        study_name=config.study_name,
        load_if_exists=config.load_if_exists,
    )
    study.set_metric_names([config.metric_name])
    return study


@app.function
def train_pytorch_trial(
    trial,
    *,
    config: OptunaStudyConfig,
    suggest_hparams,
    model_factory,
    optimizer_factory,
    dataloader_factory,
    train_fn,
    evaluate_fn,
    criterion,
    device,
) -> float:
    """Run one Optuna trial for a PyTorch supervised-learning model.

    The factories keep the generic trial runner independent of the CoverType
    architecture, optimizer, and data split while still keeping the call site
    compact in the notebook.

    Parameters
    ----------
    trial:
        Active Optuna trial.
    config:
        Trial/runtime configuration, including epoch count, TensorBoard directory,
        Optuna direction, and optional checkpoint settings.
    suggest_hparams:
        Callable that receives ``trial`` and returns a parameter dictionary.
    model_factory:
        Callable that receives the parameter dictionary and returns a PyTorch
        model.
    optimizer_factory:
        Callable that receives ``(model, params)`` and returns an optimizer.
    dataloader_factory:
        Callable that receives the parameter dictionary and returns
        ``(train_loader, val_loader, test_loader)``.
    train_fn:
        Training function compatible with ``train_model``.
    evaluate_fn:
        Evaluator passed through to ``train_fn``. It owns the primary metric name,
        value, and direction used by checkpointing and Optuna reporting.
    criterion:
        Loss function used for training and optional validation loss reporting.
    device:
        Device on which the model and batches should run.

    Returns
    -------
    float
        Best validation metric from the trial history. Optuna interprets this
        according to ``config.direction``.
    """
    params = suggest_hparams(trial)

    model = model_factory(params).to(device)
    optimizer = optimizer_factory(model, params)
    train_loader, val_loader, _ = dataloader_factory(params)

    run_name = f"trial_{trial.number:04d}"
    log_dir = str(Path(config.run_dir) / run_name)
    trial.set_user_attr("tensorboard_log_dir", log_dir)
    trial.set_user_attr("hidden_dims", list(params.get("hidden_dims", ())))

    checkpoint_dir = None
    if config.save_best_checkpoint and config.checkpoint_dir is not None:
        checkpoint_dir = str(Path(config.checkpoint_dir) / run_name)

    serializable_params = {
        key: value if isinstance(value, (bool, int, float, str)) else str(value)
        for key, value in params.items()
    }
    history, actual_log_dir = train_fn(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        epochs=config.epochs,
        device=device,
        evaluate_fn=evaluate_fn,
        run_dir=config.run_dir,
        run_name=run_name,
        show_epoch_summary=False,
        verbose=False,
        trial=trial,
        hparams=serializable_params,
        max_train_batches=config.max_train_batches,
        max_eval_batches=config.max_eval_batches,
        save_best=config.save_best_checkpoint,
        checkpoint_dir=checkpoint_dir,
        checkpoint_metadata={
            "study_name": config.study_name,
            "trial_number": trial.number,
            "params": serializable_params,
        },
    )

    expected_direction = "maximize" if history["higher_is_better"] else "minimize"
    if config.direction != expected_direction:
        raise ValueError(
            "OptunaStudyConfig.direction must match the evaluator direction: "
            f"config.direction={config.direction!r}, evaluator expects {expected_direction!r}."
        )

    best_metric = float(history["best_val_metric"])
    primary_metric = history["primary_metric"]
    trial.set_user_attr("tensorboard_log_dir", actual_log_dir)
    trial.set_user_attr("best_epoch", history["best_epoch"])
    trial.set_user_attr(f"best_val_{primary_metric}", best_metric)
    if history.get("checkpoint_path") is not None:
        trial.set_user_attr("checkpoint_path", history["checkpoint_path"])
    return best_metric


@app.function
def optimize_study(
    study,
    objective,
    config: OptunaStudyConfig,
    *,
    callbacks: tuple = (),
    stop_event=None,
):
    """Run ``study.optimize`` with explicit notebook defaults and cancellation.

    Parameters
    ----------
    study:
        Optuna study to optimize.
    objective:
        Callable that receives an Optuna trial and returns the metric to optimize.
    config:
        Study configuration supplying trial count, timeout, parallelism,
        callbacks behavior, and progress-bar settings.
    callbacks:
        Optional Optuna callbacks to run after each trial.
    stop_event:
        Optional ``threading.Event``. If set, a callback asks Optuna to stop after
        the current trial finishes.

    Returns
    -------
    optuna.Study
        The same study after optimization, returned for convenient notebook use.
    """
    optimize_callbacks = list(callbacks)
    if stop_event is not None:
        def stop_after_current_trial(running_study, frozen_trial) -> None:
            if stop_event.is_set():
                running_study.stop()

        optimize_callbacks.append(stop_after_current_trial)

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        n_jobs=config.n_jobs,
        catch=config.catch,
        callbacks=optimize_callbacks or None,
        gc_after_trial=config.gc_after_trial,
        show_progress_bar=config.show_progress_bar,
    )
    return study


@app.cell
def _(COVERTYPE_SEARCH_SPACE, DEVICE, build_dataloaders, criterion):
    def run_optuna_search(
        config: OptunaStudyConfig = OptunaStudyConfig(),
        search_space: CoverTypeSearchSpace = COVERTYPE_SEARCH_SPACE,
        *,
        callbacks: tuple = (),
        stop_event=None,
    ):
        """Run the CoverType HPO search using the generic PyTorch trial runner.

        Parameters
        ----------
        config:
            Optuna/training configuration for this notebook search. Its ``direction``
            must match the evaluator used below.
        search_space:
            CoverType hyperparameter search space. Passing this explicitly makes it
            easy to run a smaller or wider search without editing the helper.
        callbacks:
            Optional Optuna callbacks forwarded to ``optimize_study``.
        stop_event:
            Optional cancellation event used by the notebook UI to stop after the
            current trial.

        Returns
        -------
        optuna.Study
            Optimized study containing trial results and user attributes such as
            TensorBoard log paths and best epochs.
        """
        study = create_optuna_study(config)

        def build_trial_model(params: dict):
            return CovTypeModel(
                input_dim=config.input_dim,
                hidden_dims=params["hidden_dims"],
                output_dim=config.output_dim,
            )

        def build_trial_optimizer(model, params: dict):
            return optim.SGD(
                model.parameters(),
                lr=params["lr"],
                momentum=config.optimizer_momentum,
            )

        def build_trial_dataloaders(params: dict):
            return build_dataloaders(
                train_batch_size=params["batch_size"],
                eval_batch_size=config.eval_batch_size,
                test_batch_size=config.test_batch_size,
            )

        def objective(trial) -> float:
            return train_pytorch_trial(
                trial,
                config=config,
                suggest_hparams=lambda active_trial: suggest_covertype_hparams(
                    active_trial,
                    search_space,
                ),
                model_factory=build_trial_model,
                optimizer_factory=build_trial_optimizer,
                dataloader_factory=build_trial_dataloaders,
                train_fn=train_model,
                evaluate_fn=evaluate_classification_model,
                criterion=criterion,
                device=DEVICE,
            )

        return optimize_study(
            study,
            objective,
            config,
            callbacks=callbacks,
            stop_event=stop_event,
        )


    return (run_optuna_search,)


@app.cell(hide_code=True)
def _():
    optuna_stop_requested = Event()
    optuna_result = {}

    mo.md(
        "Shared Optuna state: `optuna_stop_requested` asks the study to stop "
        "after the current trial and `optuna_result` stores the latest study."
    )
    return optuna_result, optuna_stop_requested


@app.cell(column=3, hide_code=True)
def _(optuna_stop_requested):
    stop_smoke_optuna = mo.ui.button(
        label="Stop after current trial",
        kind="warn",
        on_change=lambda _: optuna_stop_requested.set(),
    )

    mo.vstack(
        [
            mo.md(
                "**Optuna smoke test.** The study runs in a background thread. "
                "Use the stop button to finish the active trial and then stop the study."
            ),
            stop_smoke_optuna,
        ]
    )
    return


@app.cell
def _(optuna_result, optuna_stop_requested, run_optuna_search):

    smoke_config = OptunaStudyConfig(
        study_name="ch10_smoke",
        storage="sqlite:///notes/optuna_smoke.db",
        n_trials=4,
        epochs=1,
        run_dir="./runs/optuna_smoke",
        max_train_batches=3,
        max_eval_batches=2,
        show_progress_bar=False,
        gc_after_trial=True,
        pruner=optuna.pruners.NopPruner(),
    )

    def optuna_smoke_loop():
        with mo.status.progress_bar(
            total=smoke_config.n_trials,
            title="Optuna smoke test",
            subtitle="Preparing study",
        ) as pbar:
            def update_progress(study, trial):
                if trial.value is None:
                    value_text = trial.state.name.lower()
                else:
                    value_text = f"value={trial.value:.4f}"

                try:
                    best_text = f" | best={study.best_value:.4f}"
                except ValueError:
                    best_text = ""

                pbar.update(
                    increment=1,
                    subtitle=f"Trial {trial.number} finished: {value_text}{best_text}",
                )

            smoke_study = run_optuna_search(
                config=smoke_config,
                callbacks=(update_progress,),
                stop_event=optuna_stop_requested,
            )

        if optuna_stop_requested.is_set():
            print("Optuna study stopped after the current trial by request.")
        else:
            print("Optuna smoke test completed the requested trials.")
        print(f"Total trials: {len(smoke_study.trials)}")
        print(f"Best value: {smoke_study.best_value:.4f}")
        print(f"Best params: {smoke_study.best_params}")

        optuna_result["config"] = smoke_config
        optuna_result["study"] = smoke_study
        optuna_result["trials"] = smoke_study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs"),
        )

    optuna_stop_requested.clear()
    optuna_result.clear()
    mo.Thread(target=optuna_smoke_loop).start()
    mo.md(
        "Optuna smoke test started in the background. Use the stop button to stop after the current trial."
    )
    return


@app.cell
def _(optuna_result):
    optuna_result
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
def _(DEVICE, HIDDEN_DIM_OPTIONS, build_dataloaders):
    # Best parameters found in optuna trials
    BEST_PARAMS = {
        "lr": 0.004108791545324082,
        "batch_size": 128,
        "hidden_dims_key": "256-512-128",
    }
    BEST_TRAIN_RUN_NAME = "best_params_sgd"

    # Initialise models and variables
    best_mod = CovTypeModel(
        input_dim=54,
        hidden_dims=HIDDEN_DIM_OPTIONS[BEST_PARAMS["hidden_dims_key"]],
        output_dim=7,
    ).to(DEVICE)

    # Optimiser
    best_optim = optim.SGD(best_mod.parameters(), momentum=0.9, lr=BEST_PARAMS["lr"])

    # Data loaders. Validation uses the larger default evaluation batch size.
    best_train_loader, best_val_loader, best_test_loader = build_dataloaders(
        train_batch_size=BEST_PARAMS["batch_size"]
    )
    return (
        BEST_TRAIN_RUN_NAME,
        best_mod,
        best_optim,
        best_train_loader,
        best_val_loader,
    )


@app.cell
def _(
    BEST_TRAIN_RUN_NAME,
    DEVICE,
    best_mod,
    best_optim,
    best_train_loader,
    best_val_loader,
    cancelled,
    criterion,
    training_result,
):
    # Decide how many additional epochs to train the model for
    epochs = 2

    def training_loop():
        run_name = training_result.setdefault("run_name", BEST_TRAIN_RUN_NAME)
        existing_history = training_result.get("history")
        start_epoch = len(existing_history["train_loss"]) if existing_history is not None else 0

        with mo.status.progress_bar(
            total=epochs * len(best_train_loader),
            title="Manual training",
            subtitle="Preparing training loop",
        ) as pbar:
            history, log_dir = train_model(
                best_mod,
                best_optim,
                criterion,
                best_train_loader,
                best_val_loader,
                epochs=epochs,
                device=DEVICE,
                evaluate_fn=evaluate_classification_model,
                run_dir="./runs/mcp_train_2506",
                run_name=run_name,
                start_epoch=start_epoch,
                save_best=True,
                history=existing_history,
                verbose=False,
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
        training_result["run_name"] = run_name


    cancelled.clear()
    mo.Thread(target=training_loop).start()
    mo.md(
        "Training started in the background. Use the stop button in the cell above to stop after the current epoch."
    )
    return


@app.cell(hide_code=True)
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


@app.cell
def _(training_result):
    # save training_results to disk
    with open(training_result["log_dir"] + "/training_result.json", "w") as f:
        json.dump(training_result, f, indent=4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
