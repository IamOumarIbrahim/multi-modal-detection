# Installs every .ttf in this folder for the current user (no admin needed).
# Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File install_fonts.ps1
# Restart PowerPoint afterwards so it sees the new fonts.
$dest = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Fonts"
$reg  = "HKCU:\Software\Microsoft\Windows NT\CurrentVersion\Fonts"
New-Item -ItemType Directory -Force $dest | Out-Null
if (-not (Test-Path $reg)) { New-Item -Path $reg -Force | Out-Null }
foreach ($f in Get-ChildItem -Path $PSScriptRoot -Filter *.ttf) {
    $target = Join-Path $dest $f.Name
    if (Test-Path $target) { Write-Host "already installed $($f.Name)"; continue }
    Copy-Item $f.FullName $target
    Set-ItemProperty -Path $reg -Name "$($f.BaseName) (TrueType)" -Value $target
    Write-Host "installed $($f.Name)"
}
