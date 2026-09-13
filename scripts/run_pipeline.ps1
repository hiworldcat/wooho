[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $PSScriptRoot
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

& $Python (Join-Path $PSScriptRoot "inspect_dataset.py")
& $Python (Join-Path $PSScriptRoot "analyze_basic_quality.py")
& $Python (Join-Path $PSScriptRoot "quality_checks.py")
& $Python (Join-Path $PSScriptRoot "image_checks.py")
& $Python (Join-Path $PSScriptRoot "multimodal_checks.py")
& $Python (Join-Path $PSScriptRoot "build_report.py")
& $Python (Join-Path $PSScriptRoot "validate_detectors.py")

Write-Host "Pipeline completed. Reports are under $root\outputs"
