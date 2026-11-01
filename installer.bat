@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Installation de MocapCam ===
where py >nul 2>nul
if errorlevel 1 (
  echo Python est introuvable. Installez Python 3.11 ou 3.12 depuis https://www.python.org
  echo en cochant "Add python.exe to PATH", puis relancez ce fichier.
  pause
  exit /b 1
)
if not exist .venv (
  py -3 -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
rem recette de compilation de l'app pour GitHub (dossier .github\workflows)
if not exist .github\workflows\app-mobile.yml (
  mkdir .github\workflows 2>nul
  copy /y mobile\workflow-github-app-mobile.yml .github\workflows\app-mobile.yml >nul
)
if not exist studio\studio.json (
  .venv\Scripts\python.exe -m mocapcam init studio
)
echo.
echo Installation terminee.
echo  - Mesurez vos boules et editez studio\studio.json
pause
