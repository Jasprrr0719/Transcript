@echo off
chcp 65001 >nul
title Meeting Transkription - Installation

echo.
echo ============================================================
echo        MEETING TRANSKRIPTION - INSTALLATION
echo ============================================================
echo.
echo Dies installiert alle noetigen Python-Pakete.
echo Das kann beim ersten Mal 5-10 Minuten dauern.
echo.
pause

echo.
echo [1/3] Pruefe Python...
python --version
if %errorlevel% neq 0 (
    echo.
    echo FEHLER: Python wurde nicht gefunden!
    echo Bitte installiere Python von https://python.org
    echo Wichtig: Beim Installieren "Add Python to PATH" anhaken!
    pause
    exit /b 1
)

echo.
echo [2/3] Aktualisiere pip...
python -m pip install --upgrade pip

echo.
echo [3/3] Installiere Pakete...
echo.

echo Installiere PyTorch mit CUDA-Support fuer NVIDIA-Karte...
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installiere restliche Pakete...
pip install faster-whisper soundcard numpy customtkinter

if %errorlevel% neq 0 (
    echo.
    echo FEHLER: Installation fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo                INSTALLATION FERTIG!
echo ============================================================
echo.
echo Du kannst die App jetzt mit start.bat starten.
echo.
pause
