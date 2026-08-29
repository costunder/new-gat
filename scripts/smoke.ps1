$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".matplotlib"
$env:PYTHONPATH = "$(Join-Path $ProjectRoot 'src');$ProjectRoot"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}

& $Python (Join-Path $ProjectRoot "scripts\run_all.py") @args
