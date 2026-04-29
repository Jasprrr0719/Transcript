@echo off
chcp 65001 >nul
title Meeting Transkription
cd /d "%~dp0"
python app.py
if %errorlevel% neq 0 (
    echo.
    echo Die App wurde mit einem Fehler beendet.
    pause
)
