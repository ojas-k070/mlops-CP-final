# 🚀 Energy Forecasting — Project Startup Guide

Quick-reference command sheet for reopening and resuming work on the **Energy Forecasting MLOps** project.

---

## ⚡ 1. Quick Reopening (TL;DR)

Whenever you open a new terminal in this project:

### Step 1: Open Terminal & Navigate to Directory
```powershell
cd "e:\CP mlops\energy-forecasting"
```

### Step 2: Activate Virtual Environment

- **PowerShell (Windows):**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # If scripts are disabled
  .\.venv\Scripts\Activate.ps1
  ```
- **Command Prompt (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Git Bash / Linux / macOS:**
  ```bash
  source .venv/Scripts/activate   # or source .venv/bin/activate
  ```

---

## 🖥️ 2. Starting the Services (Local Development)

Open 3 terminal tabs/windows (with `.venv` activated in each) and run:

### Terminal 1: FastAPI Inference Server
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **URL:** [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Terminal 2: Streamlit Analytics Dashboard
```powershell
streamlit run dashboard/app.py --server.port 8502 --server.address 0.0.0.0
```
- **Dashboard URL:** [http://localhost:8502](http://localhost:8502)

### Terminal 3: MLflow Experiment Tracking UI
```powershell
mlflow ui --port 5050 --host 0.0.0.0
```
- **MLflow Tracking UI:** [http://localhost:5050](http://localhost:5050)

---

## 🎯 3. One-Command Automated Launch

Instead of manually running 3 commands in separate terminals, use one of the automated starters:

### Option A: Windows PowerShell Script (Launches 3 separate windows)
```powershell
.\start_all.ps1
```

### Option B: Windows Batch File (Double-click or run from CMD)
```cmd
start_all.bat
```

### Option C: Git Bash / Linux Shell Script
```bash
bash run_all.sh
```

### Option D: Docker Compose (Full Stack in Container)
```bash
docker-compose up --build
```
To stop Docker services:
```bash
docker-compose down
```

---

## 🌐 4. Service Port & URL Reference Table

| Service | Port | URL | Description |
| :--- | :--- | :--- | :--- |
| **Streamlit Dashboard** | `8502` | [http://localhost:8502](http://localhost:8502) | Interactive UI for forecast visualization & controls |
| **FastAPI Backend** | `8000` | [http://localhost:8000](http://localhost:8000) | Recursive multi-step prediction API |
| **FastAPI Swagger Docs**| `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive endpoint testing & schema |
| **MLflow UI** | `5050` | [http://localhost:5050](http://localhost:5050) | Metric tracking, model artifacts & run history |

---

## 🔄 5. Data & Training Pipeline Commands

Run these when updating data or retraining the model:

```powershell
# 1. Download & preprocess raw dataset into clean hourly series
python src/data_loader.py

# 2. Train XGBoost model & log metrics/artifacts to MLflow
python src/train_model.py

# 3. Run walk-forward backtesting (outputs to reports/backtest_report.json)
python src/backtest.py

# 4. Run automated retraining workflow with Prefect
python pipelines/prefect_flow.py

# 5. Reproduce full DVC pipeline
dvc repro
```

---

## 🧪 6. Testing & Quality Gate Commands

Run tests and style/type checkers before committing code:

```powershell
# Run unit tests with pytest
pytest

# Code style check (Ruff)
ruff check src/

# Auto-fix code style issues
ruff format src/

# Strict type checking (MyPy)
mypy src/ --strict

# Or run full quality gate (if make is installed)
make verify
```

---

## 🛠️ 7. Common Troubleshooting

### 1. PowerShell Script Execution Disabled
If you see `File ... cannot be loaded because running scripts is disabled on this system`:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Port Already in Use (e.g. 8000, 8502, or 5050)
Find and kill the process occupying the port on Windows:
```powershell
# Check what process is using port 8000
netstat -ano | findstr :8000

# Kill process by PID (replace <PID> with the number in rightmost column)
taskkill /PID <PID> /F
```

### 3. Missing Dependencies
If packages are missing after pulling updates:
```powershell
pip install -r requirements.txt
# Or with uv / editable mode:
uv pip install -e ".[dev]"
```
