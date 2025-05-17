@echo off

REM Create the shortcut with better error handling
echo Creating desktop shortcut with custom icon...
powershell -Command "& {try { $WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Parking App.lnk'); $Shortcut.TargetPath='%~f0'; $Shortcut.IconLocation='%~dp0parking.ico'; $Shortcut.Save(); Write-Host 'Shortcut created successfully!' } catch { Write-Host 'Error creating shortcut: ' $_.Exception.Message }}"

echo Checking if Python is installed...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed.
    echo Please install Python from the Microsoft Store.
    echo Press any key to open the Microsoft Store...
    pause >nul
    start ms-windows-store://pdp/?ProductId=9PJPW5LDXLZ5
    echo After installing Python, please run this script again.
    pause
    exit
)

echo Installing required packages...
pip install kivy torch torchvision opencv-python ultralytics matplotlib pandas Pillow numpy 

echo Starting application...
cd /d "%~dp0APPformadmah"
python main.py
pause