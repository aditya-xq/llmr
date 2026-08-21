#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Build and package llmr releases for multiple platforms.

.DESCRIPTION
    Creates release artifacts for all supported platforms (Windows, Linux, macOS).
    Run on each platform to generate platform-specific binaries, then publish.

.PARAMETER Version
    Release version (e.g., v1.0.0). Default: v1.0.0

.PARAMETER SkipBuild
    Skip cargo build, assume binaries exist in artifacts/

.PARAMETER Publish
    Create and push GitHub release after packaging

.PARAMETER DryRun
    Show what would be done without executing

.PARAMETER Trace
    Enable trace output

.EXAMPLE
    ./release.ps1 v1.0.0

.EXAMPLE
    ./release.ps1 v1.0.0 -SkipBuild -Publish
#>

[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [ValidatePattern("^v?\d+\.\d+\.\d+")]
    [string]$Version = "v1.0.0",

    [switch]$SkipBuild,

    [switch]$Publish,

    [switch]$DryRun,

    [switch]$Trace
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-ProjectRoot {
    if ($PSScriptRoot) {
        return Split-Path -Parent $PSScriptRoot
    }
    return Split-Path -Parent $MyInvocation.PSCommandPath
}

function Write-Header($message) {
    Write-Host $message -ForegroundColor Cyan
}

function Write-Success($message) {
    Write-Host $message -ForegroundColor Green
}

function Write-Warn($message) {
    Write-Host $message -ForegroundColor Yellow
}

function Write-Step($message) {
    if ($Trace) { Write-Host "  → $message" -ForegroundColor Gray }
}

function Test-Prerequisite {
    param([string]$Command, [string]$InstallHint)

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Write-Warn "Missing: $Command"
        if ($InstallHint) { Write-Host "  Install: $InstallHint" -ForegroundColor Gray }
        return $false
    }
    return $true
}

function Get-PlatformInfo {
    if ($IsWindows) { return @{ Target = "x86_64-pc-windows-msvc"; Ext = "zip"; Binary = "llmr.exe" } }
    if ($IsMacOS) {
        $target = if ((uname -m) -match "arm64") { "aarch64-apple-darwin" } else { "x86_64-apple-darwin" }
        return @{ Target = $target; Ext = "tar.gz"; Binary = "llmr" }
    }
    if ($IsLinux) {
        $target = if ((uname -m) -match "aarch64|arm64") { "aarch64-unknown-linux-gnu" } else { "x86_64-unknown-linux-gnu" }
        return @{ Target = $target; Ext = "tar.gz"; Binary = "llmr" }
    }
    throw "Unsupported platform"
}

function Build-Binary {
    Write-Header "Building llmr..."
    $timer = [Diagnostics.Stopwatch]::StartNew()

    & cargo build --release --locked --bin llmr
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    $timer.Stop()
    Write-Step "Built in $($timer.Elapsed.TotalSeconds)s"
}

function New-Artifact {
    param(
        [string]$Target,
        [string]$Ext,
        [string]$SourcePath
    )

    $assetName = "llmr-$Target.$Ext"
    $artifactsDir = Join-Path $ProjectRoot "artifacts"

    if (-not (Test-Path $artifactsDir)) {
        New-Item -ItemType Directory -Path $artifactsDir | Out-Null
    }

    if ($Ext -eq "zip") {
        $zipPath = Join-Path $artifactsDir $assetName
        Compress-Archive -Path $SourcePath -DestinationPath $zipPath -Force
    } else {
        $tmpBin = Join-Path $artifactsDir "llmr"
        $tarPath = Join-Path $artifactsDir $assetName

        Copy-Item $SourcePath $tmpBin -Force
        tar -czf $tarPath -C $artifactsDir "llmr"
        Remove-Item $tmpBin -Force
    }

    Write-Step "Created: $assetName"
}

function Update-Checksums {
    $artifactsDir = Join-Path $ProjectRoot "artifacts"
    if (-not (Test-Path $artifactsDir)) { return }

    $lines = Get-ChildItem $artifactsDir -File |
        Where-Object { $_.Name -ne "checksums.txt" } |
        Sort-Object Name |
        ForEach-Object {
            if ($IsWindows) {
                $hash = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            } else {
                $hash = (sha256sum $_.FullName) -replace '\s.*$', ''
            }
            "$hash  $($_.Name)"
        }

    if ($lines) {
        Set-Content -Path (Join-Path $artifactsDir "checksums.txt") -Value $lines
        Write-Step "Updated: checksums.txt"
    }
}

function Copy-Installers {
    $artifactsDir = Join-Path $ProjectRoot "artifacts"

    foreach ($file in @("install.sh", "install.ps1")) {
        $src = Join-Path $ProjectRoot $file
        if (Test-Path $src) {
            Copy-Item $src $artifactsDir -Force
            Write-Step "Copied: $file"
        }
    }
}

function Get-MissingArtifacts {
    $required = @(
        "llmr-x86_64-pc-windows-msvc.zip",
        "llmr-x86_64-unknown-linux-gnu.tar.gz",
        "llmr-aarch64-unknown-linux-gnu.tar.gz",
        "llmr-x86_64-apple-darwin.tar.gz",
        "llmr-aarch64-apple-darwin.tar.gz",
        "checksums.txt",
        "install.sh",
        "install.ps1"
    )

    $artifactsDir = Join-Path $ProjectRoot "artifacts"
    $missing = @()

    foreach ($item in $required) {
        if (-not (Test-Path (Join-Path $artifactsDir $item))) {
            $missing += $item
        }
    }

    return $missing
}

$ProjectRoot = Get-ProjectRoot

if ($DryRun) {
    Write-Header "[DryRun] Release $Version"
    Write-Host "  Project: $ProjectRoot"
    Write-Host "  SkipBuild: $SkipBuild"
    Write-Host "  Publish: $Publish"
    exit 0
}

if ($Trace) { Write-Host "PowerShell $($PSVersionTable.PSVersion)" -ForegroundColor Gray }

$info = Get-PlatformInfo

Write-Header "Release $Version"
Write-Host "Target: $($info.Target)" -ForegroundColor Gray
Write-Host ""

if (-not $SkipBuild) {
    if (-not (Test-Prerequisite "cargo" "https://rustup.rs")) {
        exit 1
    }

    Build-Binary

    $binaryPath = Join-Path $ProjectRoot "target/release/$($info.Binary)"
    if (-not (Test-Path $binaryPath)) {
        throw "Binary not found: $binaryPath"
    }

    New-Artifact -Target $info.Target -Ext $info.Ext -SourcePath $binaryPath
}

Copy-Installers
Update-Checksums

$artifactsDir = Join-Path $ProjectRoot "artifacts"

Write-Header "`nArtifacts:"
if (Test-Path $artifactsDir) {
    Get-ChildItem $artifactsDir -File | ForEach-Object {
        $size = "{0:N1} KB" -f ($_.Length / 1KB)
        Write-Host "  $($_.Name.PadRight(35)) $size" -ForegroundColor Gray
    }
}
else {
    Write-Warn "  No artifacts directory found"
}

$missing = Get-MissingArtifacts
if ($missing) {
    Write-Warn "`nMissing artifacts (build on other platforms):"
    $missing | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
}

if ($Publish) {
    if (-not (Test-Prerequisite "gh" "https://cli.github.com")) {
        exit 1
    }

    Write-Header "`nPublishing to GitHub..."

    Push-Location $ProjectRoot
    try {
        $files = Get-ChildItem artifacts -File | ForEach-Object { "./artifacts/$($_.Name)" }

        gh release create $Version $files --title $Version --generate-notes

        Write-Success "`nRelease published: https://github.com/aditya-xq/llmr/releases/tag/$Version"
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Header "`nTo publish: ./scripts/release.ps1 $Version -Publish"
}