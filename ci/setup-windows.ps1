$ErrorActionPreference = 'Stop'

& "./.github/scripts/enable-graphics-apis.ps1"

New-Item -ItemType Directory -Force -Path $env:CHOCO_CACHE_LOCATION | Out-Null
choco config set cacheLocation $env:CHOCO_CACHE_LOCATION

$installer = "C:\ci-cache\installers\vc_redist.x64.exe"
New-Item -ItemType Directory -Force -Path (Split-Path $installer) | Out-Null
if (-not (Test-Path $installer)) {
  $temporary = "$installer.$env:GITHUB_RUN_ID.tmp"
  Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $temporary
  Move-Item -Force $temporary $installer
}
$signature = Get-AuthenticodeSignature $installer
if ($signature.Status -ne "Valid") { throw "Invalid VC++ redistributable signature: $($signature.Status)" }
Start-Process -FilePath $installer -ArgumentList "/install", "/quiet", "/norestart" -Wait
