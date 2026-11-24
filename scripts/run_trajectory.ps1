# Reviewer Concern 7 (trajectories, pred-vs-oracle onset, BN-only control, +ImageNet seeds)
# and Concern 3 (balanced-population lambda_t distribution at K >> B).
#   powershell -ExecutionPolicy Bypass -File scripts\run_trajectory.ps1 -MaxGpu
# Resumable (skips finished .npz). ImageNet runs need the full 1000-wnid ImageNet-C tree.
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",          # CIFAR trajectory seeds
    [string]$InSeeds = "0,1,2,3,4",    # ImageNet seeds (>=3 answers the 2-seed concern)
    [switch]$SkipCifar,
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
New-Item -ItemType Directory -Force -Path "results/trajectory" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\trajectory_$ts.log" -Append | Out-Null

function Run($desc, $cmd) {
    Write-Host "`n==================== $desc ===================="; Write-Host ">> $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] CUDA not available."; Stop-Transcript | Out-Null; exit 1 }

$ck10 = "$Data/checkpoints/resnet18_cifar10.pt"; $c10 = "$Data/cifar10c/CIFAR-10-C"

# config tuples: tag | adapter | feedback | adaptFlag | eps
$configs = @(
    @("tent_none",  "tent",        "none",   "",           "0.0"),   # baseline, no feedback
    @("tent_pred",  "tent",        "pred",   "",           "1.0"),   # collapse trajectory
    @("patta_pred", "patta_steep", "pred",   "",           "1.0"),   # rescue trajectory
    @("tent_oracle","tent",        "oracle", "",           "1.0"),   # oracle-label onset
    @("bnonly_pred","tent",        "pred",   "--no-adapt", "1.0")    # BN-only control (no grad updates)
)

# ---------------- CIFAR-10-C trajectories (gaussian_noise + contrast, sev5, bs16) ----------------
if (-not $SkipCifar) {
    $seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
    foreach ($corr in @("gaussian_noise", "contrast")) {
        foreach ($cfg in $configs) {
            foreach ($s in $seedList) {
                $out = "results/trajectory/traj_c10_${corr}_$($cfg[0])_s${s}.npz"
                if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
                Run "c10 / $corr / $($cfg[0]) / seed $s" "python -m experiments.exp_trajectory --which cifar10 --ckpt $ck10 --cdir $c10 --corruption $corr --severity 5 --eps $($cfg[4]) --bs 16 --steps 1500 --eval_every 25 --adapter $($cfg[1]) --feedback $($cfg[2]) $($cfg[3]) --p 0.5 --seed $s --out $out"
            }
        }
    }
}

# ---------------- ImageNet-C trajectories + lambda_t distribution (contrast sev3, bs16) ----------------
if (-not $SkipImagenet) {
    $incdir = "$Data/imagenet-c"
    if (Test-Path $incdir) {
        $inSeedList = $InSeeds.Split(",") | ForEach-Object { [int]$_.Trim() }
        foreach ($cfg in $configs) {
            foreach ($s in $inSeedList) {
                $out = "results/trajectory/traj_in_contrast_sev3_$($cfg[0])_s${s}.npz"
                if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
                Run "imagenet / contrast sev3 / $($cfg[0]) / seed $s" "python -m experiments.exp_trajectory --which imagenet --cdir $incdir --arch resnet50 --corruption contrast --severity 3 --eps $($cfg[4]) --bs 16 --steps 1000 --eval_every 50 --eval_bs 1000 --eval_chunk 128 --adapter $($cfg[1]) --feedback $($cfg[2]) $($cfg[3]) --p 0.5 --seed $s --out $out"
            }
        }
        # Concern 3: balanced-population lambda_t distribution at K=1000 >> B=16 (eps=0, no feedback).
        foreach ($s in $inSeedList) {
            $out = "results/trajectory/traj_in_contrast_sev3_lambda_balanced_s${s}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "imagenet / lambda_t balanced / seed $s" "python -m experiments.exp_trajectory --which imagenet --cdir $incdir --arch resnet50 --corruption contrast --severity 3 --eps 0.0 --bs 16 --steps 800 --eval_every 100 --eval_bs 1000 --eval_chunk 128 --adapter patta_steep --feedback none --p 0.5 --seed $s --out $out"
        }
    } else { Write-Host "[note] $incdir missing - ImageNet trajectories skipped." }
}

Write-Host "`nAll done. NPZ files in results\trajectory\ ; run summarize_trajectory then send the CSVs back."
Stop-Transcript | Out-Null
