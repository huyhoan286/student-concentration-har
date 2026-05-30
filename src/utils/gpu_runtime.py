"""
Tối ưu huấn luyện / inference trên GPU (CUDA).

Batch size và num_workers mặc định cho GPU mạnh; CPU/debug giữ cấu hình nhẹ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Batch gợi ý khi có CUDA (ResNet+LSTM vs MoViNet 3D)
GPU_BATCH_SIZES: dict[str, int] = {
    "lrcn": 16,
    "convlstm": 16,
    "movinet": 8,
}
CPU_BATCH_SIZE = 2


def cuda_available() -> bool:
    return torch.cuda.is_available()


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


def recommended_batch_size(model_name: str, *, debug: bool = False) -> int:
    if debug:
        return 1
    if cuda_available():
        return GPU_BATCH_SIZES.get(model_name.lower().strip(), 8)
    return CPU_BATCH_SIZE


def recommended_num_workers(*, debug: bool = False) -> int:
    if debug or not cuda_available():
        return 0
    return min(8, max(2, (os.cpu_count() or 4) // 2))


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
) -> dict:
    pin = pin_memory if pin_memory is not None else cuda_available()
    opts: dict = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        opts["persistent_workers"] = True
        opts["prefetch_factor"] = 2
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
    """Nâng batch/workers nếu đang dùng default CPU và có CUDA."""
    if config.debug:
        return config

    if not cuda_available():
        config.use_amp = False
        config.compile_model = False
        return config

    if config.batch_size <= 2:
        config.batch_size = recommended_batch_size(config.model_name, debug=False)
    if config.num_workers == 0:
        config.num_workers = recommended_num_workers(debug=False)

    return config
