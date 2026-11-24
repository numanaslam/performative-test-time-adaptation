# ENDOGENOUS-feedback run: the clean "the feedback is real" test. On a BALANCED IID stream
# (no eps, no exogenous correlation), does the model's confidence-gated replay buffer
# self-induce class imbalance and collapse? alpha = fraction of the training batch drawn
# from the self-selected buffer (0 = no feedback control, 1 = full feedback).
#   powershell -ExecutionPolicy Bypass -File scripts\run_endogenous.ps1 -MaxGpu
# Resumable (skips finished runs).
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",
    [string]$Corruptions = "gaussian_noise,contrast",
    [string]$Adapters = "source,tent,eata_lite,patta,patta_sev",
    [string]$Alphas = "0.0,0.5,1.0",
    [int]$Bs = 16,
    [int]$BufferCap = 2000,
    [int]$Steps = 2000,
    [int]$EvalEvery = 50,
    [switch]$Do100,
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
New-Item -ItemType Directory -Force -Path "results/endo" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\endo_$ts.log" -Append | Out-Null
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

$jobs = @()
$jobs += ,@("cifar10", "c10", $ck10, $c10)
if ($Do100 -and (Test-Path "$c100/labels.npy")) { $jobs += ,@("cifar100", "c100", $ck100, $c100) }

$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$corrList = $Corruptions.Split(",") | ForEach-Object { $_.Trim() }
$alphaList = $Alphas.Split(",") | ForEach-Object { $_.Trim() }

foreach ($j in $jobs) {
    $which = $j[0]; $tag = $j[1]; $ck = $j[2]; $cdir = $j[3]
    foreach ($corr in $corrList) {
        foreach ($a in $alphaList) {
            $atag = ($a -replace "\.", "")
            foreach ($seed in $seedList) {
                $out = "results/endo/endo_${tag}_${corr}_a${atag}_s${seed}.npz"
                if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
                Run "endo $tag / $corr / alpha=$a / seed $seed" "python -m experiments.exp_endogenous --which $which --ckpt $ck --cdir $cdir --corruption $corr --adapters '$Adapters' --alpha $a --buffer_cap $BufferCap --bs $Bs --steps $Steps --eval_every $EvalEvery --seed $seed --out $out"
            }
        }
    }
}

Run "endogenous summary" "python -m experiments.summarize_endogenous --glob 'results/endo/endo_*.npz' --out results/endo.csv"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Endogenous table + verdict: results\endo.csv  (and the VERDICT lines above)"
Write-Host "Full log: results\endo_$ts.log"
Stop-Transcript | Out-Null
