# Thiết lập project sau clone (Windows PowerShell)
# Usage: .\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Setup Student Concentration HAR ===" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Khong tim thay python. Cai Python 3.10+ truoc."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Tao virtualenv .venv ..."
    python -m venv .venv
}

Write-Host "Kich hoat .venv ..."
& .\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Da tao .env — sua KAGGLE_DATASET_SLUG va KAGGLE_API_TOKEN truoc khi tai data." -ForegroundColor Yellow
    Write-Host ""
}

python scripts/setup_project.py --skip-install @args

Write-Host ""
Write-Host "Kich hoat moi lan lam viec:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
