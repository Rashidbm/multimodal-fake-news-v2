# One-shot dataset build for Windows PowerShell.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_dataset.ps1
#
# Steps: HuggingFace login (asks for the token once, at the start) -> install ->
# unit tests -> download MMFakeBench -> extract every zip -> inspect (all images
# must exist) -> build balanced dataset -> verify.  Everything is written to
# run_dataset.log next to this repo so the output can be reviewed later.
# Any failing step stops the script; the log shows where.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
Start-Transcript -Path "$Root\run_dataset.log" -Append | Out-Null

function Step($name) { Write-Host "`n===== $name =====" -ForegroundColor Cyan }

Step "0. HuggingFace token (read-only; owner revokes it after the download)"
if (-not $env:HF_TOKEN) { $env:HF_TOKEN = "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" }

Step "1. Install dependencies"
pip install -r requirements.txt huggingface_hub

Step "2. Unit tests"
python -m pytest fnd/tests -q
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }

Step "3. Download MMFakeBench (resumes if interrupted)"
huggingface-cli download liuxuannan/MMFakeBench --repo-type dataset --local-dir data\raw\MMFakeBench
if ($LASTEXITCODE -ne 0) { throw "download failed" }
Get-ChildItem data\raw\MMFakeBench

Step "4. Extract every zip archive"
Get-ChildItem data\raw\MMFakeBench -Recurse -Filter *.zip | ForEach-Object {
    Write-Host "extracting $($_.FullName)"
    Expand-Archive -Path $_.FullName -DestinationPath data\raw\MMFakeBench -Force
}
Get-ChildItem data\raw\MMFakeBench -Directory -Recurse -Depth 2 | Select-Object FullName

Step "5. Inspect: map every record, require every image file"
python -m fnd.data.mmfakebench --root data\raw\MMFakeBench --require-images
if ($LASTEXITCODE -ne 0) { throw "inspect failed" }

Step "6. Build the balanced dataset and verify"
python -m fnd.data.build --mmfakebench data\raw\MMFakeBench --out data\processed --images required
if ($LASTEXITCODE -ne 0) { throw "build/verify failed" }

Step "DONE  ->  data\processed\balanced_5group.csv  (log: run_dataset.log)"
Stop-Transcript | Out-Null
