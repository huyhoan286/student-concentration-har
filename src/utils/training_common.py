"""
Tiện ích huấn luyện dùng chung cho LRCN / ConvLSTM / MoViNet.
Hỗ trợ tối ưu GPU: AMP, pin_memory, TF32, channels_last.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from constants import (
    checkpoint_path,
    training_accuracy_chart_path,
    training_log_path,
)
from gpu_runtime import (
    amp_context,
    apply_gpu_train_defaults,
    cuda_available,
    dataloader_kwargs,
    gpu_device_name,
    prepare_model_for_gpu,
    setup_gpu_runtime,
    transfer_to_device,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    model_name: str
    num_classes: int = 5
    num_frames: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 0
    debug: bool = False
    epochs: int = 5
    batch_size: int = 2
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    use_amp: bool = True
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    use_channels_last: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.debug:
            self.epochs = 1
            self.batch_size = 1
            self.max_train_samples = 10
            self.max_val_samples = 5
            self.num_workers = 0
            self.use_amp = False
            self.compile_model = False
        else:
            apply_gpu_train_defaults(self)

    @property
    def saved_model_path(self) -> Path:
        return checkpoint_path(self.model_name)

    @property
    def log_csv_path(self) -> Path:
        return training_log_path(self.model_name)

    @property
    def accuracy_chart_path(self) -> Path:
        return training_accuracy_chart_path(self.model_name)


def get_project_root() -> Path:
    return _PROJECT_ROOT


def get_device(debug: bool = False) -> torch.device:
    if debug:
        return torch.device("cpu")
    return torch.device("cuda" if cuda_available() else "cpu")


def apply_debug_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    n = min(max_samples, len(dataset))
    return Subset(dataset, list(range(n)))


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, int, int]:
    from dataset import StudentBehaviorDataset  # noqa: PLC0415

    train_ds = StudentBehaviorDataset(
        mode="train",
        project_root=_PROJECT_ROOT,
        num_frames=config.num_frames,
    )
    val_ds = StudentBehaviorDataset(
        mode="val",
        project_root=_PROJECT_ROOT,
        num_frames=config.num_frames,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Dataset train rỗng. Kiểm tra dataset/train/")
    if len(val_ds) == 0:
        raise RuntimeError("Dataset val rỗng. Kiểm tra dataset/val/")

    if config.debug:
        train_ds = apply_debug_subset(train_ds, config.max_train_samples)
        val_ds = apply_debug_subset(val_ds, config.max_val_samples)

    loader_opts = dataloader_kwargs(num_workers=config.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_opts,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_opts,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    phase: str,
    debug: bool = False,
    *,
    use_amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    amp = amp_context(device, use_amp=use_amp)
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    context = torch.enable_grad() if train else torch.inference_mode()

    with context:
        for batch_idx, (clips, labels) in enumerate(loader, start=1):
            if debug:
                print(f"{phase} batch {batch_idx}/{num_batches}")

            clips, labels = transfer_to_device(clips, labels, device)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with amp.autocast():
                logits = model(clips)
                loss = criterion(logits, labels)

            if train and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.float().item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def save_training_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_accuracy_chart(
    path: Path,
    log_rows: list[dict[str, float | int]],
    model_name: str,
    best_val_acc: float,
) -> Path | None:
    """Vẽ và lưu biểu đồ train/val accuracy theo epoch vào results/figures/."""
    if not log_rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[cảnh báo] Không có matplotlib — bỏ qua lưu biểu đồ accuracy")
        return None

    epochs = [int(row["epoch"]) for row in log_rows]
    train_acc = [float(row["train_acc"]) for row in log_rows]
    val_acc = [float(row["val_acc"]) for row in log_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, marker="o", linewidth=2, label="Train accuracy")
    ax.plot(epochs, val_acc, marker="s", linewidth=2, label="Val accuracy")

    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    ax.scatter(
        [epochs[best_idx]],
        [val_acc[best_idx]],
        color="crimson",
        s=80,
        zorder=5,
        label=f"Best val: {best_val_acc:.2f}%",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name.upper()} — Accuracy sau huấn luyện")
    ax.set_xticks(epochs)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ymin = min(min(train_acc), min(val_acc))
    ymax = max(max(train_acc), max(val_acc))
    margin = max(2.0, (ymax - ymin) * 0.05)
    ax.set_ylim(max(0.0, ymin - margin), min(100.0, ymax + margin))

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_training(
    config: TrainConfig,
    model_builder: Callable[[], nn.Module],
) -> float:
    """Huấn luyện model; trả về best val accuracy (%)."""
    mode_label = "DEBUG" if config.debug else "FULL"
    print(f"=== Train {config.model_name.upper()} [{mode_label}] ===\n")

    device = get_device(config.debug)
    if device.type == "cuda":
        setup_gpu_runtime(
            cudnn_benchmark=config.cudnn_benchmark,
            allow_tf32=config.allow_tf32,
        )

    print(f"Device: {device}", end="")
    gpu_name = gpu_device_name()
    if gpu_name:
        print(f" ({gpu_name})")
    else:
        print()

    print(
        f"Config: EPOCHS={config.epochs}, BATCH_SIZE={config.batch_size}, "
        f"LR={config.learning_rate}, NUM_FRAMES={config.num_frames}, "
        f"WORKERS={config.num_workers}, AMP={config.use_amp and device.type == 'cuda'}"
    )
    if config.compile_model and device.type == "cuda":
        print("torch.compile: bật")
    print()

    train_loader, val_loader, n_train, n_val = create_dataloaders(config)
    print(f"Train samples: {n_train}")
    print(f"Val samples:   {n_val}\n")

    model = prepare_model_for_gpu(
        model_builder(),
        device,
        use_channels_last=config.use_channels_last,
        compile_model=config.compile_model,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    config.saved_model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            phase="Train",
            debug=config.debug,
            use_amp=config.use_amp,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            phase="Val",
            debug=config.debug,
            use_amp=config.use_amp,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 4),
            }
        )

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = model.state_dict()
            if config.compile_model and hasattr(model, "_orig_mod"):
                state = model._orig_mod.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": config.model_name,
                    "model_state_dict": state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": config.num_classes,
                    "num_frames": config.num_frames,
                    "debug": config.debug,
                },
                config.saved_model_path,
            )
            print(f"  -> Lưu best model: {config.saved_model_path}")

    save_training_log(config.log_csv_path, log_rows)
    chart_path = save_training_accuracy_chart(
        config.accuracy_chart_path,
        log_rows,
        config.model_name,
        best_val_acc,
    )

    print(f"\nLog đã lưu: {config.log_csv_path}")
    if chart_path is not None:
        print(f"Biểu đồ accuracy đã lưu: {chart_path}")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Hoàn tất huấn luyện {config.model_name.upper()}.")
    return best_val_acc
