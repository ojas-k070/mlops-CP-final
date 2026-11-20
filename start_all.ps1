# Windows PowerShell Startup Script for Energy Forecasting
# Launches FastAPI, MLflow UI, and Streamlit Dashboard in separate terminal windows

$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Starting Energy Forecasting Services    " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if .venv exists
if (Test-Path "$ProjectDir\.venv\Scripts\Activate.ps1") {
    $VenvCmd = "& '$ProjectDir\.venv\Scripts\Activate.ps1';"
} else {
    $VenvCmd = ""
    Write-Warning "Virtual environment .venv not found. Using global python environment."
}

# 1. Start FastAPI API
Write-Host "[1/3] Launching FastAPI Inference Server on http://localhost:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; $VenvCmd uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 2

# 2. Start MLflow UI
Write-Host "[2/3] Launching MLflow Tracking UI on http://localhost:5050 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; $VenvCmd mlflow ui --port 5050 --host 0.0.0.0"

Start-Sleep -Seconds 2

# 3. Start Streamlit Dashboard
Write-Host "[3/3] Launching Streamlit Dashboard on http://localhost:8502 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; $VenvCmd streamlit run dashboard/app.py --server.port 8502 --server.address 0.0.0.0"

Write-Host ""
Write-Host "All 3 services launched successfully in separate windows!" -ForegroundColor Cyan
Write-Host "  - Streamlit Dashboard: http://localhost:8502" -ForegroundColor Yellow
Write-Host "  - FastAPI Docs:        http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  - MLflow UI:           http://localhost:5050" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
