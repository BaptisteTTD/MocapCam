@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Diagnostic de l'app MocapCam (Android)
set "ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
if not exist "%ADB%" (
  echo adb introuvable dans %LOCALAPPDATA%\Android\Sdk\platform-tools
  pause
  exit /b 1
)
echo Telephone branche en USB, debogage USB active ?
echo (si le telephone demande "Autoriser le debogage USB", touchez Autoriser)
echo.
"%ADB%" wait-for-device
"%ADB%" devices -l
echo.
echo Lancement de l'app MocapCam et enregistrement des erreurs...
"%ADB%" logcat -c
"%ADB%" shell monkey -p com.mocapcam.phone -c android.intent.category.LAUNCHER 1 >nul 2>nul
timeout /t 10 /nobreak >nul
"%ADB%" logcat -d -b crash > diagnostic_android_crash.txt
"%ADB%" logcat -d -v time AndroidRuntime:E ReactNativeJS:V ReactNative:V NitroModules:V VisionCamera:V CameraX:W libc:F DEBUG:F *:S > diagnostic_android_log.txt
"%ADB%" shell getprop ro.build.version.release > diagnostic_android_info.txt
"%ADB%" shell getprop ro.product.model >> diagnostic_android_info.txt
"%ADB%" shell dumpsys package com.mocapcam.phone | findstr /i "versionName targetSdk" >> diagnostic_android_info.txt
echo.
echo Termine. Fichiers crees dans %CD% :
echo   diagnostic_android_crash.txt
echo   diagnostic_android_log.txt
echo Dites a Claude "diagnostic fait".
pause
