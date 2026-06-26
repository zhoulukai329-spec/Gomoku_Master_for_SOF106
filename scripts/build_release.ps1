param(
    [string]$Version = "dev",
    [string]$WeightsPath = ""
)

$ErrorActionPreference = "Stop"

function Resolve-ReleaseWeightsPath {
    param(
        [string]$SourceDir,
        [string]$ExplicitPath
    )

    if ($ExplicitPath) {
        $resolved = Resolve-Path -Path $ExplicitPath -ErrorAction Stop
        return $resolved.Path
    }

    $weightsDir = Join-Path $SourceDir "artifacts\weights"
    if (-not (Test-Path $weightsDir)) {
        return $null
    }

    $bestCandidate = Get-ChildItem -Path $weightsDir -Filter "*_best.pth" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($bestCandidate) {
        return $bestCandidate.FullName
    }

    $latestCandidate = Get-ChildItem -Path $weightsDir -Filter "*.pth" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latestCandidate) {
        return $latestCandidate.FullName
    }

    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$srcDir = Join-Path $repoRoot "src"
$distDir = Join-Path $repoRoot "dist"
$pyInstallerWorkDir = Join-Path $repoRoot "build\pyinstaller"
$pyInstallerDistDir = Join-Path $distDir "pyinstaller"
$releaseRootDir = Join-Path $distDir "release"
$releaseName = "Gomoku_Master_$Version"
$stageDir = Join-Path $releaseRootDir $releaseName
$zipPath = Join-Path $releaseRootDir "$releaseName.zip"
$hashPath = Join-Path $releaseRootDir "$releaseName.sha256"
$exeName = "GomokuMaster.exe"

if (Test-Path $pyInstallerWorkDir) {
    Remove-Item -Path $pyInstallerWorkDir -Recurse -Force
}
if (Test-Path $pyInstallerDistDir) {
    Remove-Item -Path $pyInstallerDistDir -Recurse -Force
}
if (Test-Path $stageDir) {
    Remove-Item -Path $stageDir -Recurse -Force
}
if (Test-Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}
if (Test-Path $hashPath) {
    Remove-Item -Path $hashPath -Force
}

$weightsToBundle = Resolve-ReleaseWeightsPath -SourceDir $srcDir -ExplicitPath $WeightsPath

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name GomokuMaster `
    --distpath $pyInstallerDistDir `
    --workpath $pyInstallerWorkDir `
    --specpath $pyInstallerWorkDir `
    --paths $srcDir `
    --add-data "$srcDir\bg_china.png;." `
    "$srcDir\gui.py"

$builtExe = Join-Path $pyInstallerDistDir $exeName
if (-not (Test-Path $builtExe)) {
    throw "Build failed: $builtExe was not created."
}

New-Item -Path $stageDir -ItemType Directory -Force | Out-Null
New-Item -Path (Join-Path $stageDir "artifacts\weights") -ItemType Directory -Force | Out-Null

Copy-Item -Path $builtExe -Destination (Join-Path $stageDir $exeName)
Copy-Item -Path (Join-Path $repoRoot "README.md") -Destination (Join-Path $stageDir "README.md")

if ($weightsToBundle) {
    Copy-Item `
        -Path $weightsToBundle `
        -Destination (Join-Path $stageDir "artifacts\weights\ppo_rl_latest.pth")
    Write-Host "Bundled weights: $weightsToBundle"
} else {
    Write-Warning "No .pth file found under src\artifacts\weights. The packaged app will self-train on first launch."
}

Compress-Archive -Path $stageDir -DestinationPath $zipPath

$hash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -Path $hashPath -Value "$hash *$(Split-Path -Path $zipPath -Leaf)"

Write-Host "Release folder: $stageDir"
Write-Host "Release zip: $zipPath"
Write-Host "SHA256 file: $hashPath"
