@echo off
echo.
echo ==============================================
echo    SENTINEL DEPENDENCY REPAIR TOOL
echo ==============================================
echo.
echo Installing/Updating required libraries...
C:\Users\cherv\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install flask opencv-python ultralytics huggingface-hub pyttsx3
echo.
echo ==============================================
echo    DONE! Try running run_flask_premium.bat now.
echo ==============================================
pause
