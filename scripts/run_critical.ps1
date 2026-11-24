# CRITICAL-eps measurement: the quantitative theory<->experiment test. From the healthy
# (eps=0-adapted) model, measure the order parameter s(eps)=1-H(q)/logK across eps, per
# batch size. Theory (Appendix B) predicts: s flat then a sharp rise at eps*(B), with eps*
# INCREASING with B, and the balanced-accuracy collapse coinciding with the s-rise.
#   powershell -ExecutionPolicy Bypass -File scripts\run_critical.ps1 -MaxGpu
# Resumable (skips finished runs). CIFAR only, cheap (~20-40 min).
param(
    [string]$Data = "D:/data",
    [string]$Which = "cifar10",
    [string]$Corruptions = "gaussian_noise",
    [string]$BatchSizes = "8,16,32,64,128",
    [string]$Seeds = "0,1,2",
    [int]$Warmup = 200,
    [int]$Steps = 150,
    [string]$EpsGrid = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
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
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "results/crit" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\critical_$ts.log" -Append | Out-Null
$startTime = Get-Date

function Section($m) { Write-Host "`n==================== $m ====================" }
function Run($desc, $cmd) {
    Section $desc; Write-Host ">> $cmd"; Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

Section "PyTorch / CUDA"
python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] CUDA not available."; Stop-Transcript | Out-Null; exit 1 }

if ($Which -eq "cifar100") { $cdir = "$Data/cifar100c/CIFAR-100-C"; $ck = "$Data/checkpoints/resnet18_cifar100.pt"; $tag = "c100" }
else { $cdir = "$Data/cifar10c/CIFAR-10-C"; $ck = "$Data/checkpoints/resnet18_cifar10.pt"; $tag = "c10" }
if (-not (Test-Path "$cdir/labels.npy")) { Write-Host "[FATAL] $cdir missing."; Stop-Transcript | Out-Null; exit 1 }

$bsList   = $BatchSizes.Split(",") | ForEach-Object { [int]$_.Trim() }
$corrList = $Corruptions.Split(",") | ForEach-Object { $_.Trim() }
$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }

foreach ($corr in $corrList) {
    foreach ($b in $bsList) {
        foreach ($seed in $seedList) {
            $out = "results/crit/crit_${tag}_${corr}_bs${b}_s${seed}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "critical $tag / $corr / bs=$b / seed $seed" "python -m experiments.exp_critical --which $Which --ckpt $ck --cdir $cdir --corruption $corr --bs $b --warmup $Warmup --steps $Steps --eps_grid '$EpsGrid' --seed $seed --out $out"
        }
    }
}

Run "critical summary" "python -m experiments.summarize_critical --glob 'results/crit/crit_*.npz' --out results/critical.csv"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Bifurcation diagram + eps*(B): results\critical.csv  (and the BIFURCATION SUMMARY above)"
Write-Host "Full log: results\critical_$ts.log"
Stop-Transcript | Out-Null
