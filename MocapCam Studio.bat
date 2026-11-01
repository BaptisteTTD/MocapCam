@echo off
chcp 65001 >nul
title MocapCam Studio (laissez cette fenetre ouverte)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo MocapCam n'est pas installe : lancez d'abord installer.bat
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0"
.venv\Scripts\python.exe -m mocapcam studio
pause
