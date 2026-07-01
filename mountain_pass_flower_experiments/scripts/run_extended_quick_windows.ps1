$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_extended --data-dir data\raw --out-dir outputs_extended_quick --config configs\quick_extended.yaml --rebuild-segments --quick --skip-flower
