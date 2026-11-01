@echo off
chcp 65001 >nul
rem Glissez-deposez un dossier de prise (contenant iphone.mov et poco.mp4) sur ce fichier.
if "%~1"=="" (
  echo Glissez-deposez un dossier de prise sur ce fichier.
  pause
  exit /b 1
)
call "%~dp0mocapcam.bat" traiter "%~dp0studio" "%~1"
pause
