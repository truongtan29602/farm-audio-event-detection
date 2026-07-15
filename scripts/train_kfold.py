from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from farm_audio_event_detection.model import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpectrogramDataset(Dataset):
    """Wrap numpy arrays (N, H, W, C) → tensors (N, C, H, W)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        # X arrives as (N, 128, 250, 3); PyTorch Conv2d expects (N, C, H, W)
        self.X = torch.from_numpy(X).float().permute(0, 3, 1, 2)  # → (N, 3, 128, 250)
        y_long = torch.from_numpy(y).long()
        self.y = torch.nn.functional.one_hot(y_long, num_classes=6).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (3, 128, 250)
        if self.augment:
            x = _random_augment(x)
        return x, self.y[idx]


def _random_augment(x: torch.Tensor) -> torch.Tensor:
    """Apply lightweight spectral augmentation (SpecAugment-style).

    * Random frequency masking  (up to 15 bins)
    * Random time masking       (up to 30 frames)
    * Small Gaussian noise      (std 0.01)

    The augmentations are applied independently on each channel so that
    the three temporal resolutions stay coherent.
    """
    x = x.clone()
    _, n_mels, n_time = x.shape

    # -- frequency mask --
    f_mask = torch.randint(0, 15, (1,)).item()
    f_start = torch.randint(0, max(n_mels - f_mask, 1), (1,)).item()
    x[:, f_start : f_start + f_mask, :] = 0.0

    # -- time mask --
    t_mask = torch.randint(0, 30, (1,)).item()
    t_start = torch.randint(0, max(n_time - t_mask, 1), (1,)).item()
    x[:, :, t_start : t_start + t_mask] = 0.0

    # -- Gaussian noise --
    x = x + 0.01 * torch.randn_like(x)

    return x


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:

    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = (logits > 0.0).float()
            correct += (preds == y_batch).float().mean(dim=1).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# History plotting
# ---------------------------------------------------------------------------

def _plot_history(
    history: dict[str, list[float]],
    fold: int,
    output_path: Path,
) -> None:
    """Plot loss and accuracy curves and save to *output_path*.

    Matplotlib rcParams are set so that no text element uses bold weight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Disable bold globally
    plt.rcParams.update({
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.family": "sans-serif",
    })

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Fold {fold} training history", fontsize=13)

    # -- Loss --
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#4c72b0", linewidth=1.5, label="train")
    ax.plot(epochs, history["val_loss"],   color="#dd8452", linewidth=1.5, label="val", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Loss (fold {fold})")
    ax.legend(framealpha=0.6, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # -- Accuracy --
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color="#4c72b0", linewidth=1.5, label="train")
    ax.plot(epochs, history["val_acc"],   color="#dd8452", linewidth=1.5, label="val", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy (fold {fold})")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.6, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {output_path}")


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold: int,
    pickle_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    device: torch.device,
    num_workers: int,
) -> dict:
    """Train one fold and return the fold metrics dict."""

    print(f"\n{'='*60}")
    print(f"  Fold {fold}  —  validation fold {fold}, training folds {[f for f in range(1,6) if f != fold]}")
    print(f"{'='*60}")

    # -- Load data --
    with pickle_path.open("rb") as fh:
        data = pickle.load(fh)

    X_train: np.ndarray = data["X_train"]   # (N_train, 128, 250, 3)
    y_train: np.ndarray = data["y_train"]   # (N_train,)
    X_val:   np.ndarray = data["X_val"]     # (N_val,   128, 250, 3)
    y_val:   np.ndarray = data["y_val"]     # (N_val,)

    print(f"  Train samples : {len(y_train)}   Val samples: {len(y_val)}")

    # -- Normalize per-channel (train statistics only) --
    # X shape: (N, 128, 250, 3)  →  compute mean/std over (N, H, W) per channel
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)   # (1,1,1,3)
    std  = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std

    train_ds = SpectrogramDataset(X_train_norm, y_train, augment=True)
    val_ds   = SpectrogramDataset(X_val_norm,   y_val,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type in {"cuda", "mps"}),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type in {"cuda", "mps"}),
    )

    # -- Model, loss, optimizer, scheduler --
    model = build_model(num_classes=6, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_acc   = -1.0
    best_val_loss  = float("inf")
    checkpoint_path = output_dir / f"fold_{fold}_best.pt"

    # -- Epoch loop --
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint (primary: val acc, secondary: val loss)
        is_best = (val_acc > best_val_acc) or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "norm_mean": mean.tolist(),
                    "norm_std":  std.tolist(),
                    "class_to_index": data["class_to_index"],
                    "index_to_class": data["index_to_class"],
                },
                checkpoint_path,
            )

        # Progress line
        print(
            f"  epoch {epoch:3d}/{epochs}"
            f"  train loss {train_loss:.4f}  acc {train_acc:.4f}"
            f"  |  val loss {val_loss:.4f}  acc {val_acc:.4f}"
            + ("  *" if is_best else "")
        )

    # -- Plot --
    plot_path = output_dir / f"fold_{fold}_history.png"
    _plot_history(history, fold, plot_path)

    fold_metrics = {
        "fold": fold,
        "best_val_acc":  round(best_val_acc,  4),
        "best_val_loss": round(best_val_loss, 4),
        "final_train_acc":  round(history["train_acc"][-1],  4),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }

    print(f"\n  Best val acc for fold {fold}: {best_val_acc:.4f}")
    return fold_metrics


# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
            
    def flush(self):
        for f in self.files:
            f.flush()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FarmAudioCNN with 5-fold cross-validation."
    )
    parser.add_argument("--preprocessed-dir", default="preprocessed",
                        help="Directory containing fold_N.pkl files.")
    parser.add_argument("--output-dir", default="outputs/kfold",
                        help="Where to save checkpoints and plots.")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="auto",
                        help="'auto', 'cpu', 'cuda', or 'mps'.")
    parser.add_argument("--num-workers",  type=int,   default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -- Reproducibility --
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _select_device(args.device)
    print(f"Using device: {device}")

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = PROJECT_ROOT / "log.txt"
    log_file = log_file_path.open("w")
    sys.stdout = Tee(sys.stdout, log_file)

    all_metrics: list[dict] = []

    for fold in range(1, 6):
        pickle_path = preprocessed_dir / f"fold_{fold}.pkl"
        if not pickle_path.exists():
            print(f"[WARNING] {pickle_path} not found — skipping fold {fold}.")
            continue

        metrics = train_fold(
            fold=fold,
            pickle_path=pickle_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            device=device,
            num_workers=args.num_workers,
        )
        # Remove full history from the per-fold metrics saved in summary
        metrics_for_summary = {k: v for k, v in metrics.items() if k != "history"}
        all_metrics.append(metrics_for_summary)

    # -- Summary across folds --
    if all_metrics:
        val_accs  = [m["best_val_acc"]  for m in all_metrics]
        val_losses = [m["best_val_loss"] for m in all_metrics]

        summary = {
            "folds": all_metrics,
            "mean_val_acc":   round(float(np.mean(val_accs)),  4),
            "std_val_acc":    round(float(np.std(val_accs)),   4),
            "mean_val_loss":  round(float(np.mean(val_losses)), 4),
            "std_val_loss":   round(float(np.std(val_losses)),  4),
        }

        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"\n{'='*60}")
        print(f"  K-fold summary")
        print(f"{'='*60}")
        for m in all_metrics:
            print(f"  Fold {m['fold']}  val acc {m['best_val_acc']:.4f}  val loss {m['best_val_loss']:.4f}")
        print(f"  {'─'*40}")
        print(f"  Mean val acc : {summary['mean_val_acc']:.4f} +/- {summary['std_val_acc']:.4f}")
        print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
