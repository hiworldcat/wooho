[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReferenceRoot,

    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,

    [string]$OutputRoot,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $root "outputs\target\v2"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $PSScriptRoot
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

& $Python -X utf8 (Join-Path $root "run_v2_pipeline.py") `
    --reference-root $ReferenceRoot `
    --target-root $TargetRoot `
    --output-root $OutputRoot `
    --geometry-config (Join-Path $PSScriptRoot "geometry_config.json")

Write-Host "V2 pipeline completed. Reports are under $OutputRoot"
