param(
    [Parameter(Mandatory = $true)][string]$ReferenceRoot,
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [Parameter(Mandatory = $true)][string]$Pptx,
    [Parameter(Mandatory = $true)][string]$Pdf,
    [Parameter(Mandatory = $true)][string]$ApplicationForm,
    [Parameter(Mandatory = $true)][string]$TeamName,
    [Parameter(Mandatory = $true)][string]$ProjectName,
    [string]$OutputRoot = "outputs/target/v2",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
    & $Python "scripts/run_final_submission.py" `
        --reference-root $ReferenceRoot `
        --target-root $TargetRoot `
        --output-root $OutputRoot `
        --expected-episodes 20
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Python "scripts/prepare_submission_package.py" `
        --pptx $Pptx `
        --pdf $Pdf `
        --application-form $ApplicationForm `
        --output-root $OutputRoot `
        --expected-episodes 20 `
        --team-name $TeamName `
        --project-name $ProjectName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
