import argparse
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

from farm_audio_event_detection.model import build_model

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpectrogramDataset(Dataset):
    """Wrap numpy arrays (N, H, W, C) → tensors (N, C, H, W)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.X = torch.from_numpy(X).float().permute(0, 3, 1, 2)
        y_long = torch.from_numpy(y).long()
        self.y = torch.nn.functional.one_hot(y_long, num_classes=6).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            x = _random_augment(x)
        return x, self.y[idx]

def _random_augment(x: torch.Tensor) -> torch.Tensor:
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

def _plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Plot loss and accuracy curves and save to output_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.family": "sans-serif",
    })

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Final Model Training History", fontsize=13)

    # -- Loss --
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#4c72b0", linewidth=1.5, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss")
    ax.legend(framealpha=0.6, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # -- Accuracy --
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color="#4c72b0", linewidth=1.5, label="train acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.6, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {output_path}")

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

def main():
    parser = argparse.ArgumentParser(description="Train FarmAudioCNN on all data.")
    parser.add_argument("--preprocessed-dir", default="preprocessed")
    parser.add_argument("--output-dir", default="outputs/final_model")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--num-workers",  type=int,   default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _select_device(args.device)
    print(f"Using device: {device}")

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "train_log.txt"
    log_file = log_file_path.open("w")
    sys.stdout = Tee(sys.stdout, log_file)

    print("Loading data...")
    # Any fold has the entire dataset when combining train + val
    pickle_path = preprocessed_dir / "fold_1.pkl"
    with pickle_path.open("rb") as fh:
        data = pickle.load(fh)

    X_train_fold: np.ndarray = data["X_train"]
    y_train_fold: np.ndarray = data["y_train"]
    X_val_fold:   np.ndarray = data["X_val"]
    y_val_fold:   np.ndarray = data["y_val"]

    X_all = np.concatenate([X_train_fold, X_val_fold], axis=0)
    y_all = np.concatenate([y_train_fold, y_val_fold], axis=0)

    print(f"Total training samples: {len(y_all)}")

    mean = X_all.mean(axis=(0, 1, 2), keepdims=True)
    std  = X_all.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    X_all_norm = (X_all - mean) / std

    train_ds = SpectrogramDataset(X_all_norm, y_all, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type in {"cuda", "mps"}),
    )

    model = build_model(num_classes=6, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": []}

    print("\nStarting training on all data...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"  epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.4f}  acc {train_acc:.4f}")

    plot_path = output_dir / "final_model_history.png"
    _plot_history(history, plot_path)

    checkpoint_path = output_dir / "final_model.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "norm_mean": mean.tolist(),
            "norm_std":  std.tolist(),
            "class_to_index": data["class_to_index"],
            "index_to_class": data["index_to_class"],
        },
        checkpoint_path,
    )
    print(f"\nFinal model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
