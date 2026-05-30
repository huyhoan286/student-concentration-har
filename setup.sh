#!/usr/bin/env bash
# Thiết lập project sau clone (Linux / macOS)
# Usage: chmod +x setup.sh && ./setup.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Setup Student Concentration HAR ==="

if ! command -v python3 &>/dev/null; then
  echo "Không tìm thấy python3. Cài Python 3.10+ trước."
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Tạo virtualenv .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ""
  echo "Đã tạo .env — sửa KAGGLE_DATASET_SLUG và KAGGLE_API_TOKEN trước khi tải data."
  echo ""
fi

python scripts/setup_project.py --skip-install "$@"

echo ""
echo "Kích hoạt mỗi lần làm việc:  source .venv/bin/activate"
