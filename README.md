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
| **MoViNet-A0** | MoViNet-pytorch (Atze00) + head 5 lớp | Cần `MoViNet-pytorch` |

**Hợp đồng I/O (cả ba model):**

- **Input:** `[B, 3, T, H, W]` — mặc định `T=16`, `H=W=224`
- **Output:** `[B, 5]` logits

Factory tạo model: `src/models/model_base.py` → `build_model("lrcn" | "convlstm" | "movinet")`.

## Clone và chạy một lệnh (máy mới)

```powershell
git clone <url-repo> student-concentration-har
cd student-concentration-har

# 1. Sao chép và sửa .env (setup tự tạo từ .env.example nếu chưa có)
#    KAGGLE_DATASET_SLUG=owner/ten-bo-du-lieu
#    KAGGLE_API_TOKEN=your_token

# 2. Chạy setup (venv + pip + tải dataset + smoke test)
.\setup.ps1
```

Linux / macOS:

```bash
chmod +x setup.sh
./setup.sh
```

Script sẽ: tạo `.venv` → `pip install` → nạp `.env` → tải Kaggle (nếu thiếu `dataset/`) → test forward LRCN/ConvLSTM.

| File | Vai trò |
|------|---------|
| `.env.example` | Mẫu cấu hình (copy → `.env`) |
| `setup.ps1` / `setup.sh` | Entry point sau clone |
| `scripts/setup_project.py` | Logic setup (có thể gọi riêng) |

**Nếu đã có sẵn `dataset/`** trên máy: không cần `KAGGLE_*` — setup bỏ qua bước tải.

**Nếu đã có sẵn `dataset/`** trên máy: không cần `KAGGLE_*` — setup bỏ qua bước tải.

---

## Google Colab (GPU)

### Cách 1 — Notebook (khuyến nghị)

1. Push project lên GitHub (hoặc copy lên Google Drive).
2. Mở [Google Colab](https://colab.research.google.com/) → **File → Upload notebook**  
   hoặc **File → Open notebook → GitHub** → chọn `notebooks/train_colab.ipynb`.
3. **Runtime → Change runtime type → T4 GPU** (hoặc A100 nếu có).
4. **Secrets** 🔑 (panel trái): thêm
   - `KAGGLE_API_TOKEN`
   - `KAGGLE_DATASET_SLUG`
5. Sửa cell **CẤU HÌNH**:
   ```python
   REPO_URL = "https://github.com/<user>/student-concentration-har.git"
   USE_DRIVE = False
   MODELS_TO_TRAIN = ["lrcn", "convlstm", "movinet"]
   ```
   Hoặc `USE_DRIVE = True` và trỏ `DRIVE_PROJECT_PATH` nếu project nằm trên Drive.
6. **Run all** — notebook sẽ: clone → `colab_setup.py` → train → evaluate → copy `results/` sang Drive.

| File | Vai trò |
|------|---------|
| `notebooks/train_colab.ipynb` | Notebook chính trên Colab |
| `scripts/colab_setup.py` | Cài package, tải data, smoke test |
| `configs/colab_config.yaml` | Gợi ý cấu hình Colab |

### Cách 2 — Colab terminal (clone thủ công)

```python
!git clone https://github.com/<user>/student-concentration-har.git
%cd student-concentration-har
# Thêm Secrets KAGGLE_* rồi:
!python scripts/colab_setup.py
!python src/training/train_lrcn.py
!python src/utils/evaluate.py --model lrcn
```

**Lưu ý Colab:** session ngắn — bật `SAVE_TO_DRIVE = True` trong notebook để không mất checkpoint; `results/` không commit lên git.

**Tùy chọn:**

```powershell
.\setup.ps1 --skip-download    # chỉ cài package + smoke test
python scripts/setup_project.py --skip-install --skip-download
```

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
├── notebooks/
│   └── train_colab.ipynb       # Google Colab (GPU)
├── .env.example                # Mẫu biến môi trường (Kaggle)
├── setup.ps1 / setup.sh        # Setup một lệnh sau clone
├── scripts/
│   ├── setup_project.py
│   └── colab_setup.py          # Bootstrap Google Colab
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
| MoViNet | Cài `MoViNet-pytorch` (xem `requirements.txt`) |
| Checkpoint | Chưa có cho đến khi bạn train |

### Bước 1 — Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**MoViNet:** dùng gói [MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch) (đã có trong `requirements.txt`). `pytorchvideo` không còn export `movinet_a0`.

**Dataset từ Kaggle** (máy mới, chưa có `dataset/`):

1. `pip install kagglehub` (đã có trong `requirements.txt`)
2. Xác thực Kaggle: `kagglehub login` **hoặc** đặt `kaggle.json` tại `%USERPROFILE%\.kaggle\`
3. Sửa `configs/dataset_config.yaml`:
   ```yaml
   kaggle_dataset_slug: "owner/ten-bo-du-lieu"
   ```
4. Dataset Kaggle phải có sẵn cấu trúc `train/`, `val/`, `test/` (mỗi split chứa thư mục theo class).

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
python src/utils/test_model_forward.py --model movinet
python src/utils/test_model_forward.py --model all
```

Output đúng: `Input (2, 3, 16, 224, 224)` → `Output (2, 5)`.

### Bước 4 — Huấn luyện

```powershell
python src/training/train_lrcn.py
python src/training/train_convlstm.py
python src/training/train_movinet.py
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

**MoViNet:** cần gói `MoViNet-pytorch` (đã liệt kê trong `requirements.txt`).

**Tải dataset từ Kaggle:**

1. Xác thực: `kagglehub login` hoặc [Kaggle API token](https://www.kaggle.com/settings) → `~/.kaggle/kaggle.json`
2. Sửa `configs/dataset_config.yaml`:
   ```yaml
   kaggle_dataset_slug: "owner/ten-bo-du-lieu"
   ```
   Dataset phải **đã chia sẵn** `train/`, `val/`, `test/` (không dùng thư mục `raw`).
3. Hoặc set biến môi trường: `KAGGLE_DATASET_SLUG=owner/ten-bo-du-lieu`

3. Hoặc set biến môi trường: `KAGGLE_DATASET_SLUG=owner/ten-bo-du-lieu`

## Tối ưu GPU (tự scale theo VRAM)

Khi train, `gpu_runtime.py` **đo VRAM** và tự chọn batch/workers. In log dạng:

`VRAM 79.3 GB (high A100-class 80GB) | batch=192 workers=12 prefetch=4 compile=True`

| VRAM GPU | LRCN / ConvLSTM batch | MoViNet batch |
|----------|----------------------|---------------|
| **≥ 70 GB** (A100 80GB) | **192** | **96** |
| ≥ 40 GB | 96 | 48 |
| ≥ 20 GB | 48 | 24 |
| ≥ 10 GB (T4) | 24 | 12 |
| ≥ 6 GB | 16 | 8 |

Tự bật thêm: **AMP**, **TF32**, **pin_memory**, **prefetch=4** (VRAM ≥ 40GB), **`torch.compile`** (VRAM ≥ 40GB).

**Ép batch cao hơn** (A100 còn trống VRAM) — sửa `src/training/train_lrcn.py`:

```python
CONFIG = TrainConfig(
    model_name="lrcn",
    batch_size=256,       # ghi đè auto; tăng dần đến khi sắp OOM
    num_workers=8,
    compile_model=True,
)
```

Giá trị `batch_size` **lớn hơn** mức auto sẽ **không bị hạ xuống**.

OOM → giảm `batch_size`; lỗi `compile` → `compile_model=False`.

## Tham số huấn luyện mặc định

| Tham số | CPU / debug | GPU (tự động) |
|---------|-------------|---------------|
| `batch_size` | 2 / 1 | Tự theo VRAM (A100 80GB: **192**) |
| `num_workers` | 0 | 4–12 (VRAM lớn) |
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
| MoViNet lỗi import | `pip install git+https://github.com/Atze00/MoViNet-pytorch.git` |
| OOM GPU | Giảm `batch_size` trong `TrainConfig` hoặc bật `DEBUG` |
| Train/evaluate quá chậm | Dùng GPU; hoặc `DEBUG=True` để thử pipeline |
| `kaggle_dataset_slug` rỗng | Bình thường nếu đã có `dataset/` local; cần điền slug khi tải máy mới |
| Chữ tiếng Việt lỗi font trên terminal | Không ảnh hưởng kết quả; file log/metrics vẫn UTF-8 |

## Công nghệ chính

PyTorch, torchvision, OpenCV, scikit-learn, matplotlib, PyYAML, kagglehub, PyTorchVideo (MoViNet).

---

*Đồ án tốt nghiệp — Nhận diện hành vi & mức độ tập trung sinh viên qua video.*
