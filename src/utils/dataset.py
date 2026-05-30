"""
PyTorch Dataset đọc video từ dataset/{train|val|test} cho huấn luyện HAR.

Mỗi sample trả về:
    - clip tensor: [C, T, H, W]  (C=3, H=W=224)
    - label: int
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

Mode = Literal["train", "val", "test"]
FrameSampling = Literal["random", "sequential"]

VALID_MODES = ("train", "val", "test")
VIDEO_EXTENSION = ".mp4"
DEFAULT_NUM_FRAMES = 16
DEFAULT_FRAME_SIZE = 224


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_class_mapping(mapping_path: Path | None = None) -> dict[str, int]:
    if mapping_path is None:
        mapping_path = get_project_root() / "configs" / "class_mapping.json"

    with mapping_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    return {str(name): int(label) for name, label in raw.items()}


class StudentBehaviorDataset(Dataset):
    """
    Dataset nhận diện hành vi sinh viên từ video .mp4 trong dataset/.

    Cấu trúc:
        dataset/{train|val|test}/{class_name}/*.mp4
    """

    def __init__(
        self,
        mode: Mode = "train",
        data_root: str | Path = "dataset",
        class_to_label: dict[str, int] | None = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        frame_size: int = DEFAULT_FRAME_SIZE,
        frame_sampling: FrameSampling | None = None,
        project_root: Path | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode phải là một trong {VALID_MODES}, nhận: {mode!r}")

        self.mode = mode
        self.num_frames = num_frames
        self.frame_size = frame_size

        root = project_root or get_project_root()
        self.data_dir = (root / data_root / mode).resolve()

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {self.data_dir}")

        self.class_to_label = class_to_label or load_class_mapping()
        self.frame_sampling: FrameSampling = frame_sampling or (
            "random" if mode == "train" else "sequential"
        )

        self.samples: list[tuple[Path, int]] = self._build_sample_list()

        if len(self.samples) == 0:
            print(f"[cảnh báo] Dataset rỗng: {self.data_dir}")

    def _build_sample_list(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in self.class_to_label:
                print(f"[cảnh báo] Bỏ qua class không có trong mapping: {class_name}")
                continue

            label = self.class_to_label[class_name]
            videos = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() == VIDEO_EXTENSION
            )
            samples.extend((video_path, label) for video_path in videos)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        video_path, label = self.samples[index]
        clip = self._load_video_clip(video_path)
        return clip, label

    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        if total_frames <= 0:
            return [0] * self.num_frames

        if total_frames < self.num_frames:
            indices = list(range(total_frames))
            indices.extend([total_frames - 1] * (self.num_frames - total_frames))
            return indices

        if self.frame_sampling == "random":
            return sorted(random.sample(range(total_frames), self.num_frames))

        return np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(
            frame_rgb,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return frame_rgb.astype(np.float32) / 255.0

    def _blank_frame(self) -> np.ndarray:
        return np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32)

    def _read_frames_by_indices(
        self,
        cap: cv2.VideoCapture,
        indices: list[int],
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        last_valid: np.ndarray | None = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()

            if ret and frame_bgr is not None:
                processed = self._process_frame(frame_bgr)
                last_valid = processed
                frames.append(processed)
            elif last_valid is not None:
                frames.append(last_valid.copy())
            else:
                blank = self._blank_frame()
                last_valid = blank
                frames.append(blank)

        while len(frames) < self.num_frames:
            pad = frames[-1] if frames else self._blank_frame()
            frames.append(pad.copy())

        return frames[: self.num_frames]

    def _read_all_frames_sequential(self, cap: cv2.VideoCapture) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            frames.append(self._process_frame(frame_bgr))
        return frames

    def _load_video_clip(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Không mở được video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0:
                indices = self._sample_frame_indices(total_frames)
                frame_list = self._read_frames_by_indices(cap, indices)
            else:
                all_frames = self._read_all_frames_sequential(cap)
                if not all_frames:
                    frame_list = [self._blank_frame()] * self.num_frames
                else:
                    indices = self._sample_frame_indices(len(all_frames))
                    frame_list = [
                        all_frames[min(idx, len(all_frames) - 1)] for idx in indices
                    ]
                    while len(frame_list) < self.num_frames:
                        frame_list.append(frame_list[-1].copy())
                    frame_list = frame_list[: self.num_frames]

        finally:
            cap.release()

        clip = np.stack(frame_list, axis=0)
        return torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous().float()
