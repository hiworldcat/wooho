$ErrorActionPreference = "Stop"

$python = "C:\Users\zwd\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$root = Split-Path -Parent $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $PSScriptRoot
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

& $python -X utf8 (Join-Path $PSScriptRoot "v2_quality_pipeline.py") --geometry-config (Join-Path $PSScriptRoot "geometry_config.json")

Write-Host "V2 pipeline completed. Reports are under $root\outputs\v2"
