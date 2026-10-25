@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title NavMin Windows Launcher
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
 echo [ERROR] Python environment missing: %PY%
 echo Create it with: uv venv --python 3.13
 pause
 exit /b 1
)
:menu
cls
echo ==================================================
echo                  NAVMIN WINDOWS
echo ==================================================
echo [1] DEMO - simulated cameras and STM32 (no hardware)
echo [2] REAL - start with cameras and STM32
 echo [3] REAL - check settings and hardware prerequisites
 echo [4] DEMO - preflight only
 echo [0] Exit
 echo.
choice /c 12340 /n /m "Select [1/2/3/4/0]: "
if errorlevel 5 exit /b 0
if errorlevel 4 goto demo_check
if errorlevel 3 goto real_check
if errorlevel 2 goto real_run
if errorlevel 1 goto demo_run
:demo_run
 echo.
 echo Starting isolated demo (no serial port, no external cameras)...
 "%PY%" tools\run_diagnostic_app.py --overview localhost --stereo-left localhost --turret fake --synthetic-inputs --motion-demo
 goto done
:demo_check
 "%PY%" tools\run_diagnostic_app.py --overview localhost --stereo-left localhost --turret fake --synthetic-inputs --preflight-only
 goto done
:real_check
 "%PY%" -m navmin --preflight-only
 goto done
:real_run
 echo.
 echo WARNING: This runs production equipment. Confirm physical safety,
 echo correct calibrations and serial COM port before proceeding.
 choice /c YN /n /m "Continue with real hardware? [Y/N]: "
 if errorlevel 2 goto menu
 "%PY%" -m navmin
 goto done
:done
 echo.
 echo Exit status: %errorlevel%
 pause
 goto menu
