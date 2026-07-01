@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run scripts\setup_windows.bat first.
    exit /b 1
)
set PYTHONPATH=%CD%\src;%PYTHONPATH%
.venv\Scripts\python.exe -m mountain_pass_fl.run_all --data-dir data\raw --out-dir outputs --config configs\default.yaml --rebuild-segments
endlocal
