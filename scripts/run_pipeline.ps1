$ErrorActionPreference = "Stop"

$python = "C:\Users\zwd\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$root = Split-Path -Parent $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $PSScriptRoot
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

& $python (Join-Path $PSScriptRoot "inspect_dataset.py")
& $python (Join-Path $PSScriptRoot "analyze_basic_quality.py")
& $python (Join-Path $PSScriptRoot "quality_checks.py")
& $python (Join-Path $PSScriptRoot "image_checks.py")
& $python (Join-Path $PSScriptRoot "multimodal_checks.py")
& $python (Join-Path $PSScriptRoot "build_report.py")
& $python (Join-Path $PSScriptRoot "validate_detectors.py")

Write-Host "Pipeline completed. Reports are under $root\outputs"
