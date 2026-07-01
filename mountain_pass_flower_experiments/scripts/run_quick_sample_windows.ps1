$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
    exit 1
}

# Make src/ importable even if the editable install was not run.
$env:PYTHONPATH = "$(Join-Path $ProjectRoot 'src');$env:PYTHONPATH"

& $Python -m mountain_pass_fl.run_all `
  --data-dir "data\raw" `
  --out-dir "outputs_quick" `
  --config "configs\quick.yaml" `
  --rebuild-segments `
  --skip-flower
