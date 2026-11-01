@echo off
chcp 65001 >nul
rem Glissez-deposez un dossier de prise sur ce fichier : verifie la synchro et la detection des boules
rem (images calibration_*.jpg dans le sous-dossier mocapcam de la prise).
if "%~1"=="" (
  echo Glissez-deposez un dossier de prise sur ce fichier.
  pause
  exit /b 1
)
call "%~dp0mocapcam.bat" marqueurs "%~dp0studio" "%~1"
start "" "%~1\mocapcam"
pause
