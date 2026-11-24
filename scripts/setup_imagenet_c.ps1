# Download + extract ImageNet-C (Hendrycks & Dietterich 2019, Zenodo 2235448) for the
# SCALE experiment. Downloads only the category tarballs you ask for and extracts them
# into <Data>/imagenet-c/<corruption>/<severity>/<wnid>/*.JPEG .
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_imagenet_c.ps1 -Categories "noise,digital"
#
# Category -> corruptions it contains:
#   noise    gaussian_noise, shot_noise, impulse_noise
#   blur     defocus_blur, glass_blur, motion_blur, zoom_blur
#   weather  snow, frost, fog, brightness
#   digital  contrast, elastic_transform, pixelate, jpeg_compression
#   extra    speckle_noise, gaussian_blur, spatter, saturate
#
# NOTE: these tarballs are LARGE (~tens of GB each). Default noise+digital covers the
# {gaussian_noise, contrast} scale run. Resumable: existing <corruption>/5 dirs are skipped.
param(
    [string]$Data = "D:/data",
    [string]$Categories = "noise,digital",
    [switch]$KeepTars
)

$ErrorActionPreference = "Stop"
$root = Join-Path $Data "imagenet-c"
New-Item -ItemType Directory -Force -Path $root | Out-Null
$base = "https://zenodo.org/records/2235448/files"

# a representative corruption per category to detect an already-extracted tar (resume)
$probe = @{ noise = "gaussian_noise"; blur = "defocus_blur"; weather = "fog";
           digital = "contrast"; extra = "speckle_noise" }

$cats = $Categories.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }
Write-Host "ImageNet-C -> $root   categories: $($cats -join ', ')"
Write-Host "WARNING: each category tar is tens of GB. Ensure disk space and a stable connection.`n"

foreach ($cat in $cats) {
    if (-not $probe.ContainsKey($cat)) { Write-Host "[skip] unknown category '$cat'"; continue }
    $marker = Join-Path $root (Join-Path $probe[$cat] "5")
    if (Test-Path $marker) { Write-Host "[skip] '$cat' already extracted ($marker exists)"; continue }

    $url = "$base/$cat.tar" + "?download=1"
    $tar = Join-Path $Data "$cat.tar"
    Write-Host "==== downloading $cat.tar ===="
    Write-Host ">> curl.exe -L -o `"$tar`" `"$url`""
    curl.exe -L --fail -o "$tar" "$url"
    if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] download failed for $cat"; continue }

    Write-Host "==== extracting $cat.tar -> $root ===="
    Push-Location $root
    tar -xf "$tar"
    $ok = ($LASTEXITCODE -eq 0)
    Pop-Location
    if (-not $ok) { Write-Host "[FATAL] extract failed for $cat"; continue }
    if (-not $KeepTars) { Remove-Item "$tar" -Force -ErrorAction SilentlyContinue }
    Write-Host "[done] $cat`n"
}

Write-Host "Extracted tree check:"
Get-ChildItem $root -Directory | ForEach-Object {
    $sev5 = Join-Path $_.FullName "5"
    $nc = if (Test-Path $sev5) { (Get-ChildItem $sev5 -Directory).Count } else { 0 }
    Write-Host ("  {0,-20} severity-5 class folders: {1}" -f $_.Name, $nc)
}
Write-Host "`nFor a valid ImageNet-scale run each corruption's severity-5 must show 1000 class folders."
