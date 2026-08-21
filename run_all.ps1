param(
    [string]$Python = "python"
)

$CodeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python "$CodeDir\generate_dataset.py" --output-dir "$CodeDir\..\dataset"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$CodeDir\run_experiments.py" `
    --dataset-dir "$CodeDir\..\dataset" `
    --results-dir "$CodeDir\..\results" `
    --figures-dir "$CodeDir\..\figures"
exit $LASTEXITCODE
