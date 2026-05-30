"""
Đánh giá model HAR trên tập test.

Chạy từ thư mục gốc project:
    python src/utils/evaluate.py --model lrcn
    python src/utils/evaluate.py --model convlstm
    python src/utils/evaluate.py --model movinet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UTILS_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _PROJECT_ROOT / "src" / "models"

for path in (_UTILS_DIR, _MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constants import (  # noqa: E402
    RESULTS_FIGURES_DIR,
    RESULTS_METRICS_DIR,
    checkpoint_path,
)
from dataset import StudentBehaviorDataset  # noqa: E402
from gpu_runtime import (  # noqa: E402
    amp_context,
    cuda_available,
    dataloader_kwargs,
    gpu_device_name,
    prepare_model_for_gpu,
    recommended_batch_size,
    recommended_num_workers,
    setup_gpu_runtime,
    transfer_to_device,
)
from model_base import ACTIVE_MODELS, build_model  # noqa: E402

MAPPING_PATH = _PROJECT_ROOT / "configs" / "class_mapping.json"
METRICS_DIR = RESULTS_METRICS_DIR
FIGURES_DIR = RESULTS_FIGURES_DIR

MODEL_CONFIG = {
    "lrcn": {
        "checkpoint": checkpoint_path("lrcn"),
        "metrics_prefix": "lrcn",
        "default_num_frames": 16,
    },
    "convlstm": {
        "checkpoint": checkpoint_path("convlstm"),
        "metrics_prefix": "convlstm",
        "default_num_frames": 16,
    },
    "movinet": {
        "checkpoint": checkpoint_path("movinet"),
        "metrics_prefix": "movinet",
        "default_num_frames": 16,
    },
}

BATCH_SIZE = 2  # CPU fallback; tự nâng khi có CUDA
NUM_WORKERS = 0
USE_AMP = True
NUM_CLASSES = 5
EVAL_MODEL_CHOICES = ACTIVE_MODELS


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_mapping() -> tuple[dict[str, int], list[str]]:
    with MAPPING_PATH.open(encoding="utf-8") as f:
        class_to_label = {str(k): int(v) for k, v in json.load(f).items()}

    label_to_class = [""] * len(class_to_label)
    for name, label in class_to_label.items():
        label_to_class[label] = name
    return class_to_label, label_to_class


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_for_eval(model_name: str, device: torch.device) -> tuple[nn.Module, int]:
    """Tạo model và trả về (model, num_frames cho dataset)."""
    name = model_name.lower().strip()
    cfg = MODEL_CONFIG[name]
    path = cfg["checkpoint"]
    num_frames = cfg["default_num_frames"]

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

    checkpoint = _load_checkpoint(path, device)

    if name not in ACTIVE_MODELS:
        raise ValueError(f"Model không hỗ trợ: {model_name}")

    model = build_model(name, num_classes=NUM_CLASSES, pretrained=False)
    num_frames = int(checkpoint.get("num_frames", num_frames))

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    use_channels_last = name in ("lrcn", "convlstm")
    model = prepare_model_for_gpu(
        model,
        device,
        use_channels_last=use_channels_last,
        compile_model=False,
    )
    model.eval()
    return model, num_frames


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = True,
) -> tuple[list[int], list[int]]:
    all_labels: list[int] = []
    all_preds: list[int] = []
    amp = amp_context(device, use_amp=use_amp)

    with torch.inference_mode():
        for clips, labels in loader:
            clips, labels = transfer_to_device(clips, labels, device)
            with amp.autocast():
                logits = model(clips)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Đã lưu biểu đồ: {out_path}")
    except ImportError:
        print("[cảnh báo] Không có matplotlib — bỏ qua lưu confusion matrix PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HAR model on test set")
    parser.add_argument(
        "--model",
        type=str,
        default="lrcn",
        choices=list(EVAL_MODEL_CHOICES),
        help="Tên mô hình (mặc định: lrcn)",
    )
    return parser.parse_args()


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def main() -> None:
    _configure_stdout()
    args = parse_args()
    model_name = args.model.lower().strip()
    prefix = MODEL_CONFIG[model_name]["metrics_prefix"]

    print(f"=== Evaluate {model_name.upper()} (test set) ===\n")

    device = get_device()
    if device.type == "cuda":
        setup_gpu_runtime()
    print(f"Device: {device}", end="")
    gpu_name = gpu_device_name()
    if gpu_name:
        print(f" ({gpu_name})")
    else:
        print()

    batch_size = recommended_batch_size(model_name) if cuda_available() else BATCH_SIZE
    num_workers = recommended_num_workers()
    print(f"Batch size: {batch_size}, workers: {num_workers}\n")

    _, label_to_class = load_class_mapping()
    class_names = label_to_class

    model, num_frames = build_model_for_eval(model_name, device)
    print(f"num_frames (dataset): {num_frames}")

    test_ds = StudentBehaviorDataset(
        mode="test",
        project_root=_PROJECT_ROOT,
        num_frames=num_frames,
    )
    if len(test_ds) == 0:
        raise RuntimeError("Dataset test rỗng. Kiểm tra dataset/test/")

    print(f"Test samples: {len(test_ds)}\n")

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs(num_workers=num_workers),
    )

    y_true, y_pred = collect_predictions(
        model,
        loader,
        device,
        use_amp=USE_AMP,
    )

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (macro)")
    print(f"Recall:    {recall:.4f} (macro)")
    print(f"F1-score:  {f1:.4f} (macro)")
    print("\nClassification report:\n")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_txt = METRICS_DIR / f"{prefix}_test_metrics.txt"
    metrics_txt.write_text(
        "\n".join(
            [
                f"=== Test metrics ({model_name}) ===",
                f"Accuracy:  {acc:.4f}",
                f"Precision: {precision:.4f} (macro)",
                f"Recall:    {recall:.4f} (macro)",
                f"F1-score:  {f1:.4f} (macro)",
                "",
                "Classification report:",
                report,
                "",
                "Confusion matrix:",
                np.array2string(cm),
            ]
        ),
        encoding="utf-8",
    )

    cm_csv = METRICS_DIR / f"{prefix}_confusion_matrix.csv"
    header = ",".join(class_names)
    np.savetxt(cm_csv, cm, delimiter=",", fmt="%d", header=header, comments="")

    save_confusion_matrix_plot(
        cm,
        class_names,
        FIGURES_DIR / f"{prefix}_confusion_matrix.png",
        title=f"Confusion Matrix — {model_name.upper()} (Test)",
    )

    print(f"\nĐã lưu: {metrics_txt}")
    print(f"Đã lưu: {cm_csv}")
    print("Hoàn tất đánh giá.")


if __name__ == "__main__":
    main()
