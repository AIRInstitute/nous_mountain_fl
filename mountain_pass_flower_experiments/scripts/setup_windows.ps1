$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment .venv ..."
    py -3 -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $Python -m pip install -U pip

Write-Host "Installing requirements ..."
& $Python -m pip install -r requirements.txt

Write-Host "Installing project in editable mode ..."
& $Python -m pip install -e .

Write-Host "Done. You can now run:"
Write-Host "  .\scripts\run_quick_sample_windows.ps1"
Write-Host "  .\scripts\run_all_windows.ps1"
