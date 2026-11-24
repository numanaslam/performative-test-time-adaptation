# Reviewer Issue 1 -- feedback-generator controls (frozen-model + oracle-label feedback).
# Isolates generator-induced prior concentration from genuine adaptation collapse, at the
# CIFAR-10 collapse cell and the ImageNet-C flagship cell (contrast/sev3, gaussian/sev3).
#   powershell -ExecutionPolicy Bypass -File scripts\run_fbcontrol.ps1 -MaxGpu
# Resumable (skips finished runs). ImageNet runs need the full 1000-wnid ImageNet-C tree.
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",
    [string]$InSeeds = "0,1",
    [switch]$SkipImagenet,
    [switch]$MaxGpu
)
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
if ($MaxGpu) {
    $env:PERFTTA_GPU_MEM_FRACTION = "0.9"; $env:PERFTTA_GPU_MIN_FREE_MB = "500"
    try { (Get-Process -Id $PID).PriorityClass = 'Normal' } catch {}; Write-Host "MAX-GPU mode."
} else {
    $env:PERFTTA_GPU_MEM_FRACTION = "0.35"; $env:PERFTTA_GPU_MIN_FREE_MB = "3000"
    try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}; Write-Host "Courtesy mode."
}
$env:PERFTTA_GPU_WAIT_TIMEOUT_S = "10800"; $env:PERFTTA_GPU_POLL_S = "30"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,expandable_segments:True"

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "results/fbcontrol" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\fbcontrol_$ts.log" -Append | Out-Null

function Run($desc, $cmd) {
    Write-Host "`n==================== $desc ===================="; Write-Host ">> $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] CUDA not available."; Stop-Transcript | Out-Null; exit 1 }

$ck10 = "$Data/checkpoints/resnet18_cifar10.pt"; $c10 = "$Data/cifar10c/CIFAR-10-C"
$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }

# --- CIFAR-10 collapse cell (bs16, eps1): gaussian_noise + contrast ---
foreach ($corr in @("gaussian_noise", "contrast")) {
    foreach ($s in $seedList) {
        $out = "results/fbcontrol/fbc_c10_${corr}_bs16_s${s}.npz"
        if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
        Run "fbcontrol c10 / $corr / seed $s" "python -m experiments.exp_fbcontrol --which cifar10 --ckpt $ck10 --cdir $c10 --corruption $corr --severity 5 --eps 1.0 --bs 16 --steps 1500 --eval_every 50 --seed $s --out $out"
    }
}

# --- ImageNet-C flagship cells (contrast/sev3 clean-collapse, gaussian/sev3) ---
if (-not $SkipImagenet) {
    $incdir = "$Data/imagenet-c"
    if (Test-Path $incdir) {
        $inSeedList = $InSeeds.Split(",") | ForEach-Object { [int]$_.Trim() }
        foreach ($cs in @(@("contrast", 3), @("gaussian_noise", 3))) {
            foreach ($s in $inSeedList) {
                $out = "results/fbcontrol/fbc_in_$($cs[0])_sev$($cs[1])_s${s}.npz"
                if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
                Run "fbcontrol imagenet / $($cs[0]) sev$($cs[1]) / seed $s" "python -m experiments.exp_fbcontrol --which imagenet --cdir $incdir --arch resnet50 --corruption $($cs[0]) --severity $($cs[1]) --eps 1.0 --bs 16 --steps 1000 --eval_every 200 --eval_bs 1000 --eval_chunk 128 --seed $s --out $out"
            }
        }
    } else { Write-Host "[note] $incdir missing - ImageNet controls skipped." }
}

Write-Host "`nAll done. NPZ files in results\fbcontrol\ ; send them back for the control paragraph."
Stop-Transcript | Out-Null
