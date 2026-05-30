"""
Tối ưu huấn luyện / inference trên GPU (CUDA).

Tự scale batch/workers theo VRAM (A100 80GB → batch lớn, compile, prefetch cao).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

CPU_BATCH_SIZE = 2

# (min_vram_gb, batch_size theo model)
VRAM_BATCH_TIERS: list[tuple[float, dict[str, int]]] = [
    (70, {"lrcn": 192, "convlstm": 192, "movinet": 96}),   # A100 80GB
    (40, {"lrcn": 96, "convlstm": 96, "movinet": 48}),    # A100 40GB / A6000
    (20, {"lrcn": 48, "convlstm": 48, "movinet": 24}),    # RTX 4090 / A10
    (10, {"lrcn": 24, "convlstm": 24, "movinet": 12}),    # T4 16GB
    (6, {"lrcn": 16, "convlstm": 16, "movinet": 8}),      # T4 / RTX 3060
    (0, {"lrcn": 8, "convlstm": 8, "movinet": 4}),
]

# Giữ tương thích import cũ
GPU_BATCH_SIZES: dict[str, int] = VRAM_BATCH_TIERS[4][1]


def cuda_available() -> bool:
    return torch.cuda.is_available()


def get_gpu_vram_gb(device_index: int = 0) -> float:
    if not cuda_available():
        return 0.0
    props = torch.cuda.get_device_properties(device_index)
    return props.total_memory / (1024**3)


def get_vram_tier_name(vram_gb: float) -> str:
    if vram_gb >= 70:
        return "high (A100-class 80GB)"
    if vram_gb >= 40:
        return "large (40GB+)"
    if vram_gb >= 20:
        return "medium (20GB+)"
    if vram_gb >= 10:
        return "standard (10GB+)"
    return "compact"


def recommended_batch_size(model_name: str, *, debug: bool = False) -> int:
    if debug:
        return 1
    if not cuda_available():
        return CPU_BATCH_SIZE

    name = model_name.lower().strip()
    vram = get_gpu_vram_gb()
    for min_vram, sizes in VRAM_BATCH_TIERS:
        if vram >= min_vram:
            return sizes.get(name, sizes.get("lrcn", 16))
    return 8


def recommended_num_workers(*, debug: bool = False) -> int:
    if debug or not cuda_available():
        return 0
    cpus = os.cpu_count() or 4
    vram = get_gpu_vram_gb()
    if vram >= 40:
        return min(12, max(4, cpus - 1))
    return min(8, max(2, cpus // 2))


def recommended_prefetch_factor() -> int:
    if not cuda_available():
        return 2
    return 4 if get_gpu_vram_gb() >= 40 else 2


def should_compile_model(*, debug: bool = False) -> bool:
    return not debug and cuda_available() and get_gpu_vram_gb() >= 40


def setup_gpu_runtime(*, cudnn_benchmark: bool = True, allow_tf32: bool = True) -> None:
    """Bật cuDNN benchmark và TF32 (Ampere+) để tăng throughput."""
    if not cuda_available():
        return
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def gpu_device_name() -> str | None:
    if not cuda_available():
        return None
    return torch.cuda.get_device_name(0)


def describe_gpu_profile(model_name: str) -> str:
    if not cuda_available():
        return "CPU mode"
    vram = get_gpu_vram_gb()
    batch = recommended_batch_size(model_name)
    workers = recommended_num_workers()
    compile_on = should_compile_model()
    return (
        f"VRAM {vram:.1f} GB ({get_vram_tier_name(vram)}) | "
        f"batch={batch} workers={workers} prefetch={recommended_prefetch_factor()} "
        f"compile={compile_on}"
    )


def transfer_to_device(
    clips: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_nb = non_blocking and device.type == "cuda"
    return (
        clips.to(device, non_blocking=use_nb),
        labels.to(device, non_blocking=use_nb),
    )


def dataloader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
) -> dict:
    pin = pin_memory if pin_memory is not None else cuda_available()
    opts: dict = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        opts["persistent_workers"] = True
        opts["prefetch_factor"] = prefetch_factor or recommended_prefetch_factor()
    return opts


@dataclass
class AmpContext:
    enabled: bool
    device_type: str

    def autocast(self):
        device = "cuda" if self.device_type == "cuda" else "cpu"
        return torch.amp.autocast(device, enabled=self.enabled)


def amp_context(device: torch.device, *, use_amp: bool) -> AmpContext:
    enabled = use_amp and device.type == "cuda"
    return AmpContext(enabled=enabled, device_type=device.type)


def prepare_model_for_gpu(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_channels_last: bool = True,
    compile_model: bool = False,
) -> torch.nn.Module:
    """Đặt model lên GPU; tùy chọn channels_last (CNN frame) và torch.compile."""
    model = model.to(device)

    if device.type == "cuda" and use_channels_last and hasattr(model, "cnn"):
        model.cnn = model.cnn.to(memory_format=torch.channels_last)

    if compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def apply_gpu_train_defaults(config) -> object:
    """Scale batch/workers/compile theo VRAM GPU."""
    if config.debug:
        return config

    if not cuda_available():
        config.use_amp = False
        config.compile_model = False
        return config

    optimal_batch = recommended_batch_size(config.model_name, debug=False)

    # Nâng batch nếu đang dùng default (2) hoặc thấp hơn mức tối ưu cho GPU hiện tại
    if config.batch_size <= 2 or config.batch_size < optimal_batch:
        config.batch_size = optimal_batch

    if config.num_workers == 0:
        config.num_workers = recommended_num_workers(debug=False)

    if not config.compile_model and should_compile_model():
        config.compile_model = True

    return config
