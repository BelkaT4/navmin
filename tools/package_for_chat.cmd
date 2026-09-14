@echo off
setlocal
cd /d "%~dp0.."

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%~dp0package_for_chat.py"
) else (
    python "%~dp0package_for_chat.py"
)

echo.
pause
