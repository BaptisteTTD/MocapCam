@echo off
chcp 65001 >nul
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m mocapcam %*
