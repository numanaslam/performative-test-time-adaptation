# BASELINES: does PATTA's derived+adaptive correction beat the anti-collapse
# family (EATA, SAR, fixed-weight class balance) in the collapse regime?
# Also tightens CIFAR-100 (bs=32, where eps=0 is still healthy) to isolate
# performative harm cleanly.
# One command:  powershell -ExecutionPolicy Bypass -File scripts\run_baselines.ps1 -MaxGpu
# Resumable (skips finished runs).
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",
    [string]$Corruptions = "gaussian_noise,contrast",      # 2 reps; pass all 4 to scale up
    [string]$Adapters = "source,tent,eata_lite,sar,balance,patta",
    [int]$Bs = 16,
    [double]$Eps = 1.0,
    [int]$Steps = 1500,
    [int]$EvalEvery = 50,
    [int]$TightenBs = 32,                                   # CIFAR-100 harm-isolation batch
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
New-Item -ItemType Directory -Force -Path "results/base","results/base_harm" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\baselines_$ts.log" -Append | Out-Null
$startTime = Get-Date

function Section($m) { Write-Host "`n==================== $m ====================" }
function Run($desc, $cmd) {
    Section $desc; Write-Host ">> $cmd"; Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

Section "PyTorch / CUDA"
python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] CUDA not available."; Stop-Transcript | Out-Null; exit 1 }

$c10 = "$Data/cifar10c/CIFAR-10-C";   $ck10 = "$Data/checkpoints/resnet18_cifar10.pt"
$c100 = "$Data/cifar100c/CIFAR-100-C"; $ck100 = "$Data/checkpoints/resnet18_cifar100.pt"
if (-not (Test-Path "$c10/labels.npy")) { Write-Host "[FATAL] CIFAR-10-C missing."; Stop-Transcript | Out-Null; exit 1 }
$have100 = Test-Path "$c100/labels.npy"
if (-not $have100) { Write-Host "[note] CIFAR-100-C missing - CIFAR-100 runs skipped." }

$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$corrList = $Corruptions.Split(",") | ForEach-Object { $_.Trim() }

# datasets x batch for the MAIN comparison (all at eps=$Eps):
#   c10 @ Bs, c100 @ Bs, c100 @ TightenBs (healthier batch -> cleaner performative isolation)
$jobs = @()
$jobs += ,@("cifar10", "c10", $ck10, $c10, $Bs)
if ($have100) {
    $jobs += ,@("cifar100", "c100", $ck100, $c100, $Bs)
    $jobs += ,@("cifar100", "c100", $ck100, $c100, $TightenBs)
}

foreach ($j in $jobs) {
    $which = $j[0]; $tag = $j[1]; $ck = $j[2]; $cdir = $j[3]; $b = $j[4]
    foreach ($corr in $corrList) {
        foreach ($seed in $seedList) {
            $out = "results/base/base_${tag}_${corr}_bs${b}_s${seed}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "baselines $tag / $corr / bs=$b / seed $seed" "python -m experiments.exp_baselines --which $which --ckpt $ck --cdir $cdir --corruption $corr --adapters '$Adapters' --eps $Eps --bs $b --steps $Steps --eval_every $EvalEvery --seed $seed --out $out"
        }
    }
}

# harm isolation: CIFAR-100 @ TightenBs at eps=0 (healthy reference) vs eps=1 (above).
# compare tent/balance/patta eps=0 bal here against the eps=1 bs=TightenBs run above.
if ($have100) {
    foreach ($corr in $corrList) {
        foreach ($seed in $seedList) {
            $out = "results/base_harm/harm_c100_${corr}_bs${TightenBs}_eps0_s${seed}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "harm-ref c100 / $corr / bs=$TightenBs / eps0 / seed $seed" "python -m experiments.exp_baselines --which cifar100 --ckpt $ck100 --cdir $c100 --corruption $corr --adapters 'tent,balance,patta' --eps 0.0 --bs $TightenBs --steps $Steps --eval_every $EvalEvery --seed $seed --out $out"
        }
    }
}

Run "baseline comparison summary" "python -m experiments.summarize_baselines --glob 'results/base/base_*.npz' --out results/baselines.csv"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Comparison table: results\baselines.csv  (per dataset/corruption/bs: each adapter's balanced acc + FRAMING verdict)"
Write-Host "Harm-isolation (eps=0 ref): results\base_harm\  (compare to the eps=1 bs=$TightenBs CIFAR-100 rows)"
Write-Host "Full log: results\baselines_$ts.log"
Stop-Transcript | Out-Null
