# One-shot dataset build for Windows PowerShell.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_dataset.ps1
#
# Steps: find/install Python -> install deps -> unit tests -> download
# MMFakeBench -> extract every zip -> inspect (all images must exist) -> build
# balanced dataset -> verify.  Everything is written to run_dataset.log next to
# this repo.  Any failing step stops the script; the log shows where.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
Start-Transcript -Path "$Root\run_dataset.log" -Append | Out-Null

function Step($name) { Write-Host "`n===== $name =====" -ForegroundColor Cyan }

Step "0. Find Python"
$Py = $null
foreach ($cand in @("py -3", "python", "python3")) {
    try {
        $v = & cmd /c "$cand --version" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match "Python 3") { $Py = $cand; break }
    } catch {}
}
if (-not $Py) {
    Write-Host "Python not found; installing with winget..."
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $Py = "py -3"
}
Write-Host "using: $Py  ->  $(& cmd /c "$Py --version")"
function Python { & cmd /c "$Py $($args -join ' ')"; if ($LASTEXITCODE -ne 0) { throw "python step failed: $($args[0..1] -join ' ')" } }

Step "1. HuggingFace token (read-only; owner revokes it after the download)"
if (-not $env:HF_TOKEN) { $env:HF_TOKEN = "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" }

Step "2. Install dependencies"
Python -m pip install --upgrade pip
Python -m pip install -r requirements.txt huggingface_hub

Step "3. Unit tests"
Python -m pytest fnd/tests -q

Step "4. Download MMFakeBench (resumes if interrupted)"
Python -c "from huggingface_hub import snapshot_download; import os; snapshot_download('liuxuannan/MMFakeBench', repo_type='dataset', local_dir='data/raw/MMFakeBench', token=os.environ['HF_TOKEN'])"
Get-ChildItem data\raw\MMFakeBench

Step "5. Extract every zip archive"
Get-ChildItem data\raw\MMFakeBench -Recurse -Filter *.zip | ForEach-Object {
    Write-Host "extracting $($_.FullName)"
    Expand-Archive -Path $_.FullName -DestinationPath data\raw\MMFakeBench -Force
}
Get-ChildItem data\raw\MMFakeBench -Directory -Recurse -Depth 2 | Select-Object FullName

Step "6. Inspect: map every record, require every image file"
Python -m fnd.data.mmfakebench --root data\raw\MMFakeBench --require-images

Step "7. Build the balanced dataset and verify"
Python -m fnd.data.build --mmfakebench data\raw\MMFakeBench --out data\processed --images required

Step "DONE  ->  data\processed\balanced_5group.csv  (log: run_dataset.log)"
Stop-Transcript | Out-Null
