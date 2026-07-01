@echo off
title SAMAYA Edge AI Backend Server
echo ===================================================
echo   MEMULAI SERVER BACKEND SAMAYA (FASTAPI)
echo ===================================================
echo.

cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment .venv\Scripts\python.exe tidak ditemukan!
    echo Pastikan file bat ini berada di folder yang sama dengan folder .venv.
    pause
    exit /b
)

echo [INFO] Menjalankan main.py menggunakan python virtual environment...
.venv\Scripts\python.exe main.py

pause
