#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Bump llmr version, commit, and tag for release.

.DESCRIPTION
    Single source of truth is the [package] version in Cargo.toml.
    This script updates it, refreshes Cargo.lock, commits, and creates the
    v<version> tag that triggers the Release workflow.

.PARAMETER Version
    New version in semver format (e.g., 1.2.0). No "v" prefix.

.PARAMETER Push
    Push the commit and tag to origin after tagging.

.EXAMPLE
    ./scripts/bump.ps1 1.2.0

.EXAMPLE
    ./scripts/bump.ps1 1.2.0 -Push
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    [switch]$Push
)

$ErrorActionPreference = "Stop"

function Write-Warn($message) {
    Write-Host $message -ForegroundColor Yellow
}

$projectRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Split-Path -Parent $MyInvocation.PSCommandPath }
$cargoToml = Join-Path $projectRoot "Cargo.toml"
$tag = "v$Version"

if (-not (Test-Path $cargoToml)) {
    throw "Cargo.toml not found at $cargoToml"
}

$content = Get-Content $cargoToml -Raw
$currentVersion = if ($content -match '(?s)\[package\][^\[]*?version = "([^"]+)"') { $Matches[1] } else { throw "Could not find [package] version in Cargo.toml" }

if ($currentVersion -eq $Version) {
    Write-Warn "Cargo.toml is already at version $Version"
    return
}

$updated = $content -replace '(?s)(\[package\][^\[]*?version = )"[^"]*"', "`$1`"$Version`""
Set-Content -Path $cargoToml -Value $updated -NoNewline
Write-Host "Cargo.toml version -> $Version"

if ($WhatIfPreference) {
    Write-Host "WhatIf: skip cargo check"
}
else {
    & cargo check --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "cargo check failed; Cargo.lock may be out of sync"
    }
    Write-Host "Cargo.lock refreshed"
}

if ($PSCmdlet.ShouldProcess($projectRoot, "commit version bump and create tag $tag")) {
    git -C $projectRoot add Cargo.toml Cargo.lock
    if ($LASTEXITCODE -ne 0) { throw "git add failed" }

    git -C $projectRoot diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Warn "Nothing to commit (version unchanged?)"
        return
    }

    git -C $projectRoot commit -m "chore(release): $tag"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }

    git -C $projectRoot tag -a $tag -m "Release $tag"
    if ($LASTEXITCODE -ne 0) { throw "git tag $tag failed (already exists?)" }

    Write-Host "Created commit and tag $tag"
}

if ($Push) {
    if ($PSCmdlet.ShouldProcess("origin", "push $tag")) {
        git -C $projectRoot push origin HEAD
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
        git -C $projectRoot push origin $tag
        if ($LASTEXITCODE -ne 0) { throw "git push $tag failed" }
        Write-Host "Pushed; Release workflow will start: https://github.com/aditya-xq/llmr/actions"
    }
}
else {
    Write-Host "`nNext: git push origin HEAD && git push origin $tag"
}
