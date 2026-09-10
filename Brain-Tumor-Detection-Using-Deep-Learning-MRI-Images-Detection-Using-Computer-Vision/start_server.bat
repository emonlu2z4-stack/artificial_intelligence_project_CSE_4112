@echo off
title NeuroScan AI - Medical MRI Tumor Detection Server
color 0b

echo ======================================================================
echo           NeuroScan AI - Precision MRI Diagnostic System
echo ======================================================================
echo.
echo [1/3] Navigating to application directory...
cd /d "%~dp0"

echo [2/3] Checking if Port 5000 is occupied...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do (
    echo [INFO] Terminating old process on port 5000 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

echo [3/3] Launching Python Flask AI Engine on http://127.0.0.1:5000...
echo.
echo ----------------------------------------------------------------------
echo  Server running! Keep this terminal open during use.
echo  Access interface at: http://127.0.0.1:5000
echo ----------------------------------------------------------------------
echo.

start "" http://127.0.0.1:5000

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with an error code.
    pause
)