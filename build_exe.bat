@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo   Building Spacoust.exe  (takes 2-5 minutes)
echo ================================================

python -m pip install numpy scipy pillow opencv-python openai customtkinter pyinstaller
if errorlevel 1 goto :fail

set EXTRA=
if exist api_key.txt set EXTRA=--add-data "api_key.txt;."

python -m PyInstaller --noconfirm --clean --onefile --windowed --name Spacoust ^
  --icon icon.ico ^
  --add-data "logo.png;." ^
  --collect-data customtkinter ^
  --hidden-import PIL._tkinter_finder ^
  %EXTRA% ^
  desktop_app.py
if errorlevel 1 goto :fail

echo.
echo ================================================
echo   DONE!  Your app:  dist\Spacoust.exe
echo   One file - send it to anyone, the key is inside.
echo ================================================
explorer dist
pause
exit /b 0

:fail
echo.
echo   Build failed - copy the error above and send it.
pause
exit /b 1
