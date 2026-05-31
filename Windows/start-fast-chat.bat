@echo off
title Portable AI - Fast Web Chat
color 0B

echo ===================================================
echo     Portable AI - Fast Web Chat Mode
echo ===================================================
echo.
echo  Launches the AI engine + browser chat UI.
echo  All chats auto-save to the USB drive.
echo.

:: Set paths to USB Shared Folder
set "OLLAMA_MODELS=%~dp0..\Shared\models\ollama_data"
set "OLLAMA_ORIGINS=*"
set "OLLAMA_HOST=127.0.0.1:11434"

:: -------------------------------------------------------
:: Find Python: prefer portable USB copy, then system
:: -------------------------------------------------------
set "PYTHON_CMD="

:: Check for portable Python bundled on USB
if exist "%~dp0..\Shared\python\python.exe" (
    set "PYTHON_CMD=%~dp0..\Shared\python\python.exe"
    echo [OK] Using portable Python from USB drive.
    goto :PythonReady
)

:: Check for system-installed Python
python --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=python"
    echo [OK] Using system Python.
    goto :PythonReady
)

:: Python not found anywhere
echo ===================================================
echo  ERROR: Python not found!
echo ===================================================
echo.
echo  Downloading portable Python to USB drive...
echo  (This only happens once, ~11MB download)
echo.

curl -L "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" -o "%~dp0..\Shared\python-embed.zip"
if %errorlevel% neq 0 (
    echo Download failed. Please check your internet connection.
    pause
    exit
)

echo Extracting...
powershell -Command "Expand-Archive -Path '%~dp0..\Shared\python-embed.zip' -DestinationPath '%~dp0..\Shared\python' -Force"
del "%~dp0..\Shared\python-embed.zip" >nul 2>&1

if exist "%~dp0..\Shared\python\python.exe" (
    set "PYTHON_CMD=%~dp0..\Shared\python\python.exe"
    echo [OK] Portable Python installed on USB successfully!
) else (
    echo Failed to extract Python. Please try again.
    pause
    exit
)

:: -------------------------------------------------------
:: Start Ollama Engine
:: -------------------------------------------------------
:PythonReady

:: Check if Ollama is already running
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Ollama is already running - using existing instance.
    goto :StartChat
)

:: Try portable engine first
if exist "%~dp0..\Shared\bin\ollama-windows.exe" (
    echo Starting portable Ollama engine...
    start /b "" "%~dp0..\Shared\bin\ollama-windows.exe" serve
    echo Waiting for engine to initialize...
    :WaitLoop
    timeout /t 1 /nobreak >nul
    curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 goto :WaitLoop
    echo [OK] Engine is online!
    goto :StartChat
)

:: Fall back to system Ollama
where ollama >nul 2>&1
if %errorlevel%==0 (
    echo Portable engine not found - starting system Ollama...
    start /b "" ollama serve
    echo Waiting for engine to initialize...
    :WaitLoopSys
    timeout /t 1 /nobreak >nul
    curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 goto :WaitLoopSys
    echo [OK] System Ollama is online!
    goto :StartChat
)

echo.
echo ===================================================
echo  ERROR: No Ollama engine found!
echo ===================================================
echo.
echo  No Ollama engine is running and none was found on
echo  this system. To fix this, either:
echo    1. Run "install.bat" to download the portable
echo       engine, OR
echo    2. Install Ollama from https://ollama.com and
echo       make sure it is running before starting.
echo.
pause
exit

:: -------------------------------------------------------
:: Start Chat Server
:: -------------------------------------------------------
:StartChat
echo.
echo ===================================================
echo  AI ENGINE IS RUNNING
echo  Chat UI opening at: http://localhost:3333
echo  Close this window to shut down everything.
echo ===================================================
echo.

%PYTHON_CMD% "%~dp0..\Shared\chat_server.py"

echo Shutting down...
taskkill /f /im ollama-windows.exe >nul 2>&1
echo Done. Goodbye!
pause
