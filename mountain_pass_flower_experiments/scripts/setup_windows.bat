@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
)
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
echo Done.
echo Run scripts\run_quick_sample_windows.bat or scripts\run_all_windows.bat
endlocal
