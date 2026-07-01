$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_extended --data-dir data\raw --out-dir outputs_extended --config configs\extended.yaml --rebuild-segments --skip-flower
