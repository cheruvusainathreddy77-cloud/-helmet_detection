@echo off
setlocal enabledelayedexpansion

title Sentinel AI | Premium Launcher
mode con: cols=100 lines=30
color 0b

echo.
echo   ########################################################################################
echo   #                                                                                      #
echo   #   SSSSS  EEEEE  N   N  TTTTT  IIIII  N   N  EEEEE  L       V     V   22222           #
echo   #   S      E      NN  N    T      I    NN  N  E      L       V     V  2     2          #
echo   #   SSSSS  EEEE   N N N    T      I    N N N  EEEE   L       V     V      22           #
echo   #       S  E      N  NN    T      I    N  NN  E      L        V   V     22             #
echo   #   SSSSS  EEEEE  N   N    T    IIIII  N   N  EEEEE  LLLLL     V V    222222           #
echo   #                                                                                      #
echo   #                      ADVANCED HELMET DETECTION ENGINE v2.0                           #
echo   ########################################################################################
echo.

:: --- CONFIGURATION ---
set "PYTHON_EXE=python"
:: Check if the user has that specific python path from previous scripts
if exist "C:\Users\cherv\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=C:\Users\cherv\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

echo [STEP 1/3] Verifying System Architecture...
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+ and add it to PATH.
    pause
    exit /b
)
echo [OK] Python detected.

echo.
echo [STEP 2/3] Validating Neural Network Dependencies...
echo This may take a moment on first run...
%PYTHON_EXE% -c "import fastapi, uvicorn, ultralytics, cv2, pyttsx3" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Missing dependencies detected. Launching Auto-Repair...
    %PYTHON_EXE% -m pip install fastapi uvicorn jinja2 opencv-python ultralytics huggingface-hub pyttsx3 --quiet
    echo [OK] Dependencies installed successfully.
) else (
    echo [OK] All core modules are ready.
)

echo.
echo [STEP 3/3] Initializing Sentinel AI Engine...
echo.
echo -------------------------------------------------------------------
echo    SERVER STARTING AT: http://localhost:8000
echo    KEEP THIS WINDOW OPEN WHILE USING THE SYSTEM
echo -------------------------------------------------------------------
echo.

%PYTHON_EXE% web_server.py

if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL] Server crashed or was unable to start.
    echo Checking for common issues...
    %PYTHON_EXE% -m pip install ultralytics --upgrade
    echo Retrying...
    %PYTHON_EXE% web_server.py
)

pause
