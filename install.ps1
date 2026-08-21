# llmr Windows installer
# Usage:
#   irm https://raw.githubusercontent.com/aditya-xq/llmr/main/install.ps1 | iex
#   .\install.ps1 -DryRun

param(
    [switch]$DryRun,
    [switch]$CheckOnly,
    [string]$Version = $env:VERSION,
    [string]$InstallDir = "$env:LOCALAPPDATA\llmr\bin"
)

$ErrorActionPreference = "Stop"

$GitHubRepo = "aditya-xq/llmr"
$BinaryName = "llmr"
$StateDir = "$env:APPDATA\$BinaryName"
$UseColor = [string]::IsNullOrWhiteSpace($env:NO_COLOR)
$Esc = [char]27

if ($UseColor) {
    $Amber = "$Esc[38;5;214m"
    $Terracotta = "$Esc[38;5;166m"
    $Sage = "$Esc[38;5;108m"
    $Slate = "$Esc[38;5;245m"
    $Cream = "$Esc[38;5;229m"
    $Burnt = "$Esc[38;5;208m"
    $Bold = "$Esc[1m"
    $Reset = "$Esc[0m"
} else {
    $Amber = ""; $Terracotta = ""; $Sage = ""; $Slate = ""; $Cream = ""; $Burnt = ""
    $Bold = ""; $Reset = ""
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = "latest"
}

foreach ($arg in $args) {
    switch ($arg) {
        "--dry-run" { $DryRun = $true }
        "--check-only" { $CheckOnly = $true }
        default {
            Write-Host "Unknown option: $arg" -ForegroundColor Red
            exit 2
        }
    }
}

function Paint {
    param([string]$Color, [string]$Text)
    return "$Color$Text$Reset"
}

function Info { param([string]$Message) Write-Host "  $(Paint $Amber '->') $Message" }
function Ok { param([string]$Message) Write-Host "  $(Paint $Sage 'OK') $Message" }
function Warn { param([string]$Message) Write-Host "  $(Paint "$Bold$Burnt" '!') $Message" }
function Err { param([string]$Message) Write-Host "  $(Paint "$Bold$Terracotta" 'X') $Message" }
function Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "  $(Paint $Amber '->') $(Paint "$Bold$Cream" $Message)"
}
function Dry { param([string]$Message) Write-Host "    $(Paint $Slate 'dry-run') $Message" }
function Kv {
    param([string]$Key, [string]$Value)
    Write-Host "    $(Paint $Slate $Key) $(Paint $Cream $Value)"
}

function Get-CurrentVersion {
    param([string]$Binary)
    if (-not (Test-Path $Binary)) {
        return "0.0.0"
    }

    try {
        $output = & $Binary version 2>&1 | Out-String
        if ($output -match '(\d+\.\d+\.\d+)') {
            return $matches[1]
        }
    } catch {}

    return "0.0.0"
}

function Get-LatestVersion {
    if ($Version -ne "latest") {
        return $Version
    }
    if ($DryRun) {
        return "latest"
    }

    try {
        $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$GitHubRepo/releases/latest" -UseBasicParsing
        return ($response.tag_name -replace '^v', '')
    } catch {
        Err "Could not determine latest release version"
        exit 1
    }
}

function Test-VersionGte {
    param([string]$Left, [string]$Right)
    if (($Left -notmatch '^\d+\.\d+\.\d+$') -or ($Right -notmatch '^\d+\.\d+\.\d+$')) {
        return $false
    }
    try {
        return ([version]$Left -ge [version]$Right)
    } catch {
        return $false
    }
}

function Get-DownloadUrl {
    param([string]$ReleaseVersion)
    if ($ReleaseVersion -eq "latest") {
        return "https://github.com/$GitHubRepo/releases/latest/download/$BinaryName-x86_64-pc-windows-msvc.zip"
    }
    return "https://github.com/$GitHubRepo/releases/download/v$ReleaseVersion/$BinaryName-x86_64-pc-windows-msvc.zip"
}

function Test-ArchiveChecksum {
    param([string]$ArchivePath, [string]$ReleaseVersion)

    $base = "https://github.com/$GitHubRepo/releases"
    $sumsUrl = if ($ReleaseVersion -eq "latest") {
        "$base/latest/download/checksums.txt"
    } else {
        "$base/download/v$ReleaseVersion/checksums.txt"
    }

    $asset = "$BinaryName-x86_64-pc-windows-msvc.zip"
    try {
        $sums = (Invoke-WebRequest -Uri $sumsUrl -UseBasicParsing).Content
    } catch {
        Warn "Could not fetch checksums.txt ($($_.Exception.Message)) - skipping verification"
        return
    }

    $expected = ($sums -split "`n") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_.EndsWith($asset) } |
        ForEach-Object { ($_ -split '\s+')[0] } |
        Select-Object -First 1

    if (-not $expected) {
        throw "checksums.txt has no entry for ${asset} - refusing to install"
    }

    $actual = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected.ToLowerInvariant()) {
        throw "Checksum mismatch for ${asset}`n  expected: $expected`n  actual:   $actual"
    }
    Ok "Checksum verified"
}

function Install-Binary {
    param([string]$ReleaseVersion)

    $url = Get-DownloadUrl -ReleaseVersion $ReleaseVersion
    $tempFile = Join-Path $env:TEMP "$BinaryName-$ReleaseVersion.zip"
    $exePath = Join-Path $InstallDir "$BinaryName.exe"
    $displayVersion = if ($ReleaseVersion -eq "latest") { "latest" } else { "v$ReleaseVersion" }

    Step "Install $BinaryName $displayVersion"
    Kv "target      " "x86_64-pc-windows-msvc"
    Kv "install dir " $InstallDir
    Kv "download    " $url

    if ($DryRun) {
        Dry "create $InstallDir and $StateDir"
        Dry "download release archive"
        Dry "extract $BinaryName.exe into $InstallDir"
        Dry "write $StateDir\version"
        return
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

    Invoke-WebRequest -Uri $url -OutFile $tempFile -UseBasicParsing
    Test-ArchiveChecksum -ArchivePath $tempFile -ReleaseVersion $ReleaseVersion
    Expand-Archive -Path $tempFile -DestinationPath $InstallDir -Force
    Remove-Item $tempFile -Force

    $found = Get-ChildItem $InstallDir -Filter "$BinaryName*.exe" | Select-Object -First 1
    if ($found -and $found.FullName -ne $exePath) {
        Move-Item $found.FullName $exePath -Force
    }

    if (-not (Test-Path $exePath)) {
        throw "Installed binary was not found at $exePath"
    }

    $ReleaseVersion | Out-File (Join-Path $StateDir "version") -Encoding UTF8
    Ok "Installed $BinaryName v$ReleaseVersion"
}

function Test-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Warn "Docker was not found"
        return 2
    }

    try {
        docker info 2>$null | Out-Null
        Ok "Docker is running"
        return 0
    } catch {
        Warn "Docker is installed but not running"
        return 1
    }
}

function Show-DockerHint {
    if ($DryRun) {
        Dry "show Docker Desktop install instructions"
        return
    }
    Info "Docker Desktop: https://www.docker.com/products/docker-desktop/"
    Start-Process "https://www.docker.com/products/docker-desktop/"
}

function Test-Python {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
    if (-not $python) {
        $python = Get-Command python -ErrorAction SilentlyContinue
    }

    if (-not $python) {
        Warn "Python was not found"
        return 2
    }

    try {
        $py = $python.Source
        $pyVersion = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        Ok "Python $pyVersion is installed"
    } catch {
        Ok "Python is installed"
    }
    return 0
}

function Show-PythonHint {
    if ($DryRun) {
        Dry "show Python install instructions"
        return
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Info "Install Python with: winget install Python.Python.3.12"
    } else {
        Info "Python downloads: https://www.python.org/downloads/"
        Start-Process "https://www.python.org/downloads/"
    }
}

function Add-ToPath {
    $current = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($current -like "*$InstallDir*") {
        return
    }

    if ($DryRun) {
        Dry "add $InstallDir to the user PATH"
        return
    }

    [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$current", "User")
    Info "Added $InstallDir to PATH"
    Info "Restart your terminal for PATH changes to take effect"
}

function Validate-Install {
    param([string]$ExePath)
    if ($DryRun) {
        Dry "run $ExePath version"
        return
    }

    $installed = Get-CurrentVersion -Binary $ExePath
    if ($installed -eq "0.0.0") {
        throw "Could not validate $BinaryName at $ExePath"
    }
    Ok "$BinaryName v$installed is ready"
}

$exePath = Join-Path $InstallDir "$BinaryName.exe"

Write-Host ""
Write-Host "  $(Paint "$Bold$Amber" $BinaryName) $(Paint $Cream 'installer')"
Write-Host "  $(Paint $Slate 'repo:') $GitHubRepo"
if ($DryRun) {
    Warn "Dry run mode: no changes will be made"
}
Write-Host ""

$currentVersion = Get-CurrentVersion -Binary $exePath
$latestVersion = Get-LatestVersion

Kv "current     " $currentVersion
Kv "latest      " $latestVersion

if ($CheckOnly) {
    if (($currentVersion -ne "0.0.0") -and (Test-VersionGte $currentVersion $latestVersion)) {
        Ok "Current version is up to date"
    } else {
        Warn "Update available: $latestVersion"
    }
    exit 0
}

if (($currentVersion -eq "0.0.0") -or (-not (Test-VersionGte $currentVersion $latestVersion))) {
    Install-Binary -ReleaseVersion $latestVersion
} else {
    Ok "Already on latest version"
}

Step "Check dependencies"
$dockerStatus = Test-Docker
if ($dockerStatus -eq 2) {
    Show-DockerHint
}

$pythonStatus = Test-Python
if ($pythonStatus -eq 2) {
    Show-PythonHint
}

Step "Finish"
Validate-Install -ExePath $exePath
Add-ToPath

Write-Host ""
Ok "Done"
Info "Run: $(Paint $Cream "$BinaryName serve")"
