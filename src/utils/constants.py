from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"


def checkpoint_path(model_name: str) -> Path:
    return RESULTS_CHECKPOINTS_DIR / f"{model_name}_best.pth"


def training_log_path(model_name: str) -> Path:
    return RESULTS_LOGS_DIR / f"{model_name}_training_log.csv"


def training_accuracy_chart_path(model_name: str) -> Path:
    return RESULTS_FIGURES_DIR / f"{model_name}_training_accuracy.png"


CLASSES = [
    "normal",
    "distracted",
    "sleep",
    "use_smartphone",
    "drink_eat",
]

NUM_CLASSES = 5

# Mô hình chính của project
ACTIVE_MODELS = ["lrcn", "convlstm", "movinet"]

DEFAULT_NUM_FRAMES = {
    "lrcn": 16,
    "convlstm": 16,
    "movinet": 16,
}
