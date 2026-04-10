<#
.SYNOPSIS
  Trust your managed-network / SSL-inspection CA for Python, pip, requests, and gdown.

.DESCRIPTION
  TRY THIS FIRST (no PEM file needed if IT deployed the CA into the Windows store):
    pip install truststore
    .\.venv\Scripts\python.exe scripts\test_https.py
    .\.venv\Scripts\python.exe scripts\run_boxmot.py eval --benchmark mot17-ablation --tracker boosttrack --verbose
  Or: .\scripts\Invoke-Boxmot.ps1 eval --benchmark mot17-ablation --tracker boosttrack --verbose

  If SSL still fails, use a PEM bundle:
  1) Place one or more PEM files from IT (root/intermediate used to sign the proxy cert),
     OR export from Windows cert store (Base-64 X.509 .cer then rename to .pem if PEM).
  2) Run this script to merge them with certifi and optionally persist user env vars.

.PARAMETER CorporateCaPem
  Path(s) to PEM certificate file(s). Repeat -CorporateCaPem for multiple files.

.PARAMETER OutputBundle
  Where to write the merged bundle (default: repo/ssl/cacert-combined.pem).

.PARAMETER PersistUserEnvironment
  If set, sets SSL_CERT_FILE, REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE for your Windows user
  so new terminals inherit them.

.PARAMETER VenvPython
  Python to run merge script (default: ..\.venv\Scripts\python.exe if present).

.EXAMPLE
  .\scripts\Configure-PythonSSL.ps1 -CorporateCaPem "$env:USERPROFILE\certs\org-root.pem" -PersistUserEnvironment
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$CorporateCaPem,

    [string]$OutputBundle = "",

    [switch]$PersistUserEnvironment,

    [string]$VenvPython = ""
)

$ErrorActionPreference = "Stop"
# PSScriptRoot = ...\person-tracking-project\scripts  ->  repo root is one level up
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }

if (-not $OutputBundle) {
    $OutputBundle = Join-Path $RepoRoot "ssl\cacert-combined.pem"
}

if (-not $VenvPython) {
    $candidate = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $candidate) { $VenvPython = $candidate }
    else { $VenvPython = "python" }
}

$mergeScript = Join-Path $PSScriptRoot "merge_python_ssl_bundle.py"
$argList = @($mergeScript, "-o", $OutputBundle) + $CorporateCaPem
& $VenvPython @argList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$bundlePath = (Resolve-Path $OutputBundle).Path

# Current session (BoxMOT, pip, gdown in this terminal)
$env:SSL_CERT_FILE = $bundlePath
$env:REQUESTS_CA_BUNDLE = $bundlePath
$env:CURL_CA_BUNDLE = $bundlePath

Write-Host "Set for this session: SSL_CERT_FILE, REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE -> $bundlePath"

if ($PersistUserEnvironment) {
    foreach ($name in @("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")) {
        [Environment]::SetEnvironmentVariable($name, $bundlePath, "User")
    }
    Write-Host "Persisted the same three variables to your Windows user environment (new terminals)."
}

Write-Host @"

Next steps:
  - Open a NEW terminal if you used -PersistUserEnvironment.
  - Recommended for BoxMOT on Windows: also use UTF-8 mode:
      `$env:PYTHONUTF8 = '1'
  - Run: boxmot eval --benchmark mot17-ablation --tracker boosttrack --verbose

If downloads still fail, confirm with IT that you have the ROOT that signs the proxy certificate,
not only the leaf site cert.
"@
