# Student Concentration HAR

Hệ thống **nhận diện hành vi sinh viên trong lớp học** từ video (Human Activity Recognition — HAR), phục vụ đánh giá mức độ tập trung. Dự án so sánh ba kiến trúc deep learning trên cùng bộ dữ liệu và pipeline huấn luyện thống nhất.

## Mục tiêu

- Phân loại clip video ngắn vào **5 lớp hành vi**
- Huấn luyện và đánh giá ba model: **LRCN**, **ConvLSTM**, **MoViNet-A0**
- Báo cáo metric trên tập **test** (accuracy, precision, recall, F1, confusion matrix)

## Nhãn (5 lớp)

| Lớp | Mô tả ngắn |
|-----|------------|
| `normal` | Ngồi học / tập trung bình thường |
| `distracted` | Mất tập trung (không ngủ, không dùng điện thoại) |
| `sleep` | Ngủ gật |
| `use_smartphone` | Sử dụng điện thoại |
| `drink_eat` | Uống nước / ăn |

Ánh xạ nhãn: `configs/class_mapping.json`.

## Mô hình

| Model | Kiến trúc (tóm tắt) | Ghi chú |
|-------|---------------------|---------|
| **LRCN** | ResNet18 (từng frame) + LSTM + FC | Backbone ImageNet |
| **ConvLSTM** | ResNet18 + LSTM + FC | Cấu trúc tương tự LRCN, triển khai riêng |
| **MoViNet-A0** | PyTorchVideo `movinet_a0` + head 5 lớp | Cần `pytorchvideo` |

**Hợp đồng I/O (cả ba model):**

- **Input:** `[B, 3, T, H, W]` — mặc định `T=16`, `H=W=224`
- **Output:** `[B, 5]` logits

Factory tạo model: `src/models/model_base.py` → `build_model("lrcn" | "convlstm" | "movinet")`.

## Cấu trúc thư mục

```
student-concentration-har/
├── configs/
│   ├── class_mapping.json      # Tên lớp → id
│   └── dataset_config.yaml     # Đường dẫn data, slug Kaggle, hyperparams data
├── dataset/                    # Không commit (gitignore)
│   ├── train/{class}/*.mp4
│   ├── val/{class}/*.mp4
│   └── test/{class}/*.mp4
├── results/                    # Không commit
│   ├── checkpoints/            # Sau train: {model}_best.pth
│   ├── logs/                   # CSV loss/acc từng epoch
│   ├── metrics/                # Sau evaluate: test metrics, confusion matrix CSV
│   └── figures/                # Biểu đồ accuracy (train) + confusion matrix (evaluate)
├── src/
│   ├── models/
│   │   ├── model_base.py
│   │   ├── lrcn.py
│   │   ├── convlstm.py
│   │   └── movinet.py
│   ├── training/
│   │   ├── train_lrcn.py
│   │   ├── train_convlstm.py
│   │   └── train_movinet.py
│   └── utils/
│       ├── constants.py
│       ├── dataset.py            # StudentBehaviorDataset
│       ├── download_dataset.py   # Tải Kaggle → dataset/
│       ├── training_common.py    # Loop train/val, AMP, lưu checkpoint
    ├── gpu_runtime.py        # Batch/workers GPU, TF32, DataLoader
│       ├── test_model_forward.py # Smoke test (chưa train)
│       └── evaluate.py           # Đánh giá checkpoint trên test set
├── requirements.txt
└── README.md
```

## Hướng dẫn chạy (từng bước)

> Chạy **mọi lệnh** từ thư mục gốc `student-concentration-har/`.

### Bước 0 — Kiểm tra nhanh môi trường

```powershell
cd c:\DoAnTotNghiep\student-concentration-har
python --version          # cần 3.10+
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

| Thành phần | Trạng thái kiểm tra |
|------------|---------------------|
| Dataset local | OK — 1028 video (`train` 715 / `val` 207 / `test` 106) |
| LRCN forward | OK |
| ConvLSTM forward | OK |
| MoViNet | Cần cài `pytorchvideo` (xem bước 1) |
| Checkpoint | Chưa có cho đến khi bạn train |

### Bước 1 — Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**MoViNet (tùy chọn):** nếu `pip install pytorchvideo` lỗi trên Windows, vẫn train/evaluate được **LRCN** và **ConvLSTM**.

**Dataset từ Kaggle** (máy mới, chưa có `dataset/`):

1. Tải `kaggle.json` từ [Kaggle Settings](https://www.kaggle.com/settings) → đặt tại `%USERPROFILE%\.kaggle\kaggle.json`
2. Sửa `configs/dataset_config.yaml`:
   ```yaml
   kaggle_dataset_slug: "owner/ten-bo-du-lieu"
   ```
3. Dataset Kaggle phải có sẵn cấu trúc `train/`, `val/`, `test/` (mỗi split chứa thư mục theo class).

### Bước 2 — Chuẩn bị dữ liệu

```powershell
python src/utils/download_dataset.py --check-only   # kiểm tra
python src/utils/download_dataset.py                # tải nếu thiếu
```

Kết quả mong đợi: thư mục `dataset/train|val|test/{normal,distracted,sleep,use_smartphone,drink_eat}/*.mp4`.

### Bước 3 — Smoke test (chưa cần train)

Kiểm tra model build và forward với tensor giả:

```powershell
python src/utils/test_model_forward.py --model lrcn
python src/utils/test_model_forward.py --model convlstm
python src/utils/test_model_forward.py --model movinet    # cần pytorchvideo
python src/utils/test_model_forward.py --model all
```

Output đúng: `Input (2, 3, 16, 224, 224)` → `Output (2, 5)`.

### Bước 4 — Huấn luyện

```powershell
python src/training/train_lrcn.py
python src/training/train_convlstm.py
python src/training/train_movinet.py      # cần pytorchvideo
```

**Mặc định:** 5 epoch, batch 2, 16 frame, lr `1e-4`. Model **best theo val accuracy** được lưu tự động.

**Thử nhanh (debug):** mở file train tương ứng, đặt `DEBUG = True` → 1 epoch, 10 mẫu train / 5 mẫu val, chạy CPU.

**Khuyến nghị:** train trên **GPU** (Colab/Kaggle Notebook) nếu máy local không có CUDA — full train trên CPU có thể rất lâu.

Sau train, kiểm tra:

```powershell
dir results\checkpoints\    # {model}_best.pth
dir results\logs\           # {model}_training_log.csv
dir results\figures\        # {model}_training_accuracy.png
```

### Bước 5 — Đánh giá trên tập test

Cần checkpoint tương ứng trong `results/checkpoints/`:

```powershell
python src/utils/evaluate.py --model lrcn
python src/utils/evaluate.py --model convlstm
python src/utils/evaluate.py --model movinet
```

Lưu ý: evaluate đọc **106 video test** — trên CPU có thể mất **vài phút đến >10 phút** tùy máy.

Sau evaluate:

```powershell
dir results\metrics\    # *_test_metrics.txt, *_confusion_matrix.csv
dir results\figures\    # *_confusion_matrix.png
```

### Bước 6 — Dùng checkpoint cho inference thực tế

Checkpoint (`results/checkpoints/{model}_best.pth`) chứa trọng số đã học. Để dự đoán video mới:

1. `build_model("lrcn", num_classes=5, pretrained=False)`
2. `model.load_state_dict(checkpoint["model_state_dict"])`
3. Preprocess giống train: 16 frame, 224×224, tensor `[1, 3, 16, 224, 224]`
4. `logits = model(x)` → `pred = logits.argmax(dim=1)`

*(Project hiện có pipeline train/evaluate; demo camera/API cần viết thêm script inference riêng.)*

---

## Quy trình tóm tắt

```
[Cài đặt] → [Dataset] → (Smoke test) → [Train] → [Evaluate]
                ↓                         ↓            ↓
           dataset/              results/checkpoints/   results/metrics/
                                 results/logs/         results/figures/
                                 results/figures/
```

## Cài đặt (chi tiết)

Yêu cầu: **Python 3.10+**, khuyến nghị GPU (CUDA) khi train đầy đủ.

```bash
cd student-concentration-har
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

**MoViNet:** cần gói `pytorchvideo` (đã liệt kê trong `requirements.txt`). Nếu cài lỗi trên Windows, có thể train/evaluate trước hai model LRCN và ConvLSTM.

**Tải dataset từ Kaggle:**

1. Tạo API token: [Kaggle Settings](https://www.kaggle.com/settings) → đặt `kaggle.json` tại `~/.kaggle/` (hoặc `%USERPROFILE%\.kaggle\` trên Windows).
2. Sửa `configs/dataset_config.yaml`:
   ```yaml
   kaggle_dataset_slug: "owner/ten-bo-du-lieu"
   ```
   Dataset phải **đã chia sẵn** `train/`, `val/`, `test/` (không dùng thư mục `raw`).
3. Hoặc set biến môi trường: `KAGGLE_DATASET_SLUG=owner/ten-bo-du-lieu`

3. Hoặc set biến môi trường: `KAGGLE_DATASET_SLUG=owner/ten-bo-du-lieu`

## Tối ưu GPU

Khi có **CUDA**, project tự bật (xem `src/utils/gpu_runtime.py`):

| Tối ưu | Mô tả |
|--------|--------|
| **Batch size** | LRCN/ConvLSTM: 16 — MoViNet: 8 (CPU: 2) |
| **AMP (FP16)** | Mixed precision train/eval |
| **DataLoader** | `pin_memory`, `persistent_workers`, `prefetch_factor` |
| **TF32 + cuDNN benchmark** | Ampere+ / GPU NVIDIA |
| **channels_last** | ResNet18 frame encoder (LRCN, ConvLSTM) |
| **AdamW** | Optimizer thay Adam |
| **non_blocking** | Copy tensor CPU→GPU song song |

Tùy chỉnh trong `TrainConfig` (`training_common.py`):

```python
TrainConfig(
    model_name="lrcn",
    batch_size=32,        # ghi đè auto nếu > 2
    compile_model=True,   # torch.compile (PyTorch 2+, thử trên Linux)
    use_amp=True,
)
```

OOM → giảm `batch_size`; tắt `compile_model` nếu lỗi khi lưu/load checkpoint.

## Tham số huấn luyện mặc định

| Tham số | CPU / debug | GPU (tự động) |
|---------|-------------|---------------|
| `batch_size` | 2 / 1 | LRCN & ConvLSTM: 16 — MoViNet: 8 |
| `num_workers` | 0 | 2–8 (theo CPU) |
| `use_amp` | False | True |
| `epochs` | 5 | 5 |
| `num_frames` | 16 | 16 |
| `learning_rate` | 1e-4 | 1e-4 |

## Kết quả lưu trữ

| Giai đoạn | Đường dẫn | Nội dung |
|-----------|-----------|----------|
| **Train** | `results/checkpoints/{model}_best.pth` | `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_acc`, `num_frames`, … |
| **Train** | `results/logs/{model}_training_log.csv` | `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc` |
| **Train** | `results/figures/{model}_training_accuracy.png` | Biểu đồ train/val accuracy theo epoch |
| **Evaluate** | `results/metrics/{model}_test_metrics.txt` | Accuracy, precision/recall/F1 (macro), classification report |
| **Evaluate** | `results/metrics/{model}_confusion_matrix.csv` | Ma trận nhầm lẫn |
| **Evaluate** | `results/figures/{model}_confusion_matrix.png` | Heatmap (cần `matplotlib`) |

`{model}` ∈ `lrcn`, `convlstm`, `movinet`.

## Cấu hình

| File | Vai trò |
|------|---------|
| `configs/dataset_config.yaml` | `data_dir`, `kaggle_dataset_slug`, danh sách `classes`, `image_size`, `num_frames` |
| `configs/class_mapping.json` | Map tên lớp → chỉ số nhãn (0–4) |

## Luồng dữ liệu (tóm tắt)

```
Video .mp4  →  StudentBehaviorDataset  →  [B,3,16,224,224]
                                              ↓
                                    LRCN / ConvLSTM / MoViNet
                                              ↓
                                    logits [B, 5]  →  argmax → nhãn
```

- **Train/val:** dùng khi gọi script `train_*.py`
- **Test:** chỉ dùng khi gọi `evaluate.py` (không tham gia chọn checkpoint)

## Xử lý lỗi thường gặp

| Triệu chứng | Hướng xử lý |
|-------------|-------------|
| `Dataset train rỗng` | Chạy `download_dataset.py` hoặc kiểm tra `dataset/train/{class}/` |
| `Không tìm thấy checkpoint` | Train model trước; kiểm tra `results/checkpoints/` |
| MoViNet / `pytorchvideo` lỗi import | `pip install pytorchvideo` hoặc chỉ dùng LRCN + ConvLSTM |
| OOM GPU | Giảm `batch_size` trong `TrainConfig` hoặc bật `DEBUG` |
| Train/evaluate quá chậm | Dùng GPU; hoặc `DEBUG=True` để thử pipeline |
| `kaggle_dataset_slug` rỗng | Bình thường nếu đã có `dataset/` local; cần điền slug khi tải máy mới |
| Chữ tiếng Việt lỗi font trên terminal | Không ảnh hưởng kết quả; file log/metrics vẫn UTF-8 |

## Công nghệ chính

PyTorch, torchvision, OpenCV, scikit-learn, matplotlib, PyYAML, Kaggle API, PyTorchVideo (MoViNet).

---

*Đồ án tốt nghiệp — Nhận diện hành vi & mức độ tập trung sinh viên qua video.*
