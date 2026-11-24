# Download CIFAR-10-C and CIFAR-100-C to D:\data on the GPU box (Windows 10+).
# Windows ships curl.exe and tar.exe, so no extra tools are needed.
#   powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
$ErrorActionPreference = "Stop"
$data = "D:\data"
New-Item -ItemType Directory -Force -Path "$data\cifar10c", "$data\cifar100c", "$data\checkpoints", "$data\cifar" | Out-Null

function Get-IfMissing($url, $out) {
    if (Test-Path $out) { Write-Host "exists: $out" }
    else { Write-Host "downloading -> $out"; curl.exe -L --fail -C - -o $out $url }
}

Get-IfMissing "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"  "$data\cifar10c\CIFAR-10-C.tar"
Get-IfMissing "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1" "$data\cifar100c\CIFAR-100-C.tar"

Write-Host "extracting (this creates CIFAR-10-C\ and CIFAR-100-C\ subfolders of .npy files)..."
tar.exe -xf "$data\cifar10c\CIFAR-10-C.tar"   -C "$data\cifar10c"
tar.exe -xf "$data\cifar100c\CIFAR-100-C.tar" -C "$data\cifar100c"

Write-Host "done."
Write-Host "  CIFAR-10-C  pool: $data\cifar10c\CIFAR-10-C"
Write-Host "  CIFAR-100-C pool: $data\cifar100c\CIFAR-100-C"
Write-Host "Clean CIFAR-10/100 auto-download on first training run into $data\cifar"
