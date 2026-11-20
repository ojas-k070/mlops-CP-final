@echo off
echo ==========================================
echo   Starting Energy Forecasting Services
echo ==========================================

cd /d "%~dp0"

echo [1/3] Starting FastAPI on http://localhost:8000 ...
start "Energy Forecasting - FastAPI" cmd /k "call .venv\Scripts\activate.bat && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo [2/3] Starting MLflow UI on http://localhost:5050 ...
start "Energy Forecasting - MLflow" cmd /k "call .venv\Scripts\activate.bat && mlflow ui --port 5050 --host 0.0.0.0"

timeout /t 2 /nobreak >nul

echo [3/3] Starting Streamlit Dashboard on http://localhost:8502 ...
start "Energy Forecasting - Streamlit" cmd /k "call .venv\Scripts\activate.bat && streamlit run dashboard/app.py --server.port 8502 --server.address 0.0.0.0"

echo.
echo All services launched!
echo - Streamlit Dashboard: http://localhost:8502
echo - FastAPI Docs:        http://localhost:8000/docs
echo - MLflow UI:           http://localhost:5050
echo ==========================================
