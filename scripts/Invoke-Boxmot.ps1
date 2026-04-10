<#
.SYNOPSIS
  Run BoxMOT with Windows cert store (truststore) + UTF-8 mode. Passes all args through.

.EXAMPLE
  .\scripts\Invoke-Boxmot.ps1 eval --benchmark mot17-ablation --tracker boosttrack --verbose
#>
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$venvScripts = Join-Path $RepoRoot ".venv\Scripts"
$py = Join-Path $venvScripts "python.exe"
$launcher = Join-Path $PSScriptRoot "run_boxmot.py"
if (-not (Test-Path $py)) {
    Write-Error "Activate or create .venv first: $py not found"
}
# So subprocess can find `uv` when BoxMOT runs `uv pip install` (WinError 2 if missing).
$env:PATH = "$venvScripts;$env:PATH"
$env:PYTHONUTF8 = "1"
& $py $launcher @args
