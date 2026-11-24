# NATURAL-feedback run: does the performative collapse arise WITHOUT the injected eps knob,
# from a realistic correlated stream + the adapter's own confidence selection (what EATA/SAR
# already do)? This answers the biggest reviewer risk ("is the feedback real?").
#   powershell -ExecutionPolicy Bypass -File scripts\run_natural.ps1 -MaxGpu
# Resumable (skips finished runs).
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",
    [string]$Corruptions = "gaussian_noise,contrast",
    [string]$Adapters = "source,tent,eata_lite,sar,patta,patta_sev",
    [string]$Rhos = "0.0,0.9,0.99",      # 0 = iid control; ->1 = more correlated (dose-response)
    [int]$Bs = 16,
    [int]$Steps = 1500,
    [int]$EvalEvery = 50,
    [switch]$Do100,                      # also run CIFAR-100 (optional)
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
New-Item -ItemType Directory -Force -Path "results/natural" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\natural_$ts.log" -Append | Out-Null
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
$rhoList  = $Rhos.Split(",") | ForEach-Object { $_.Trim() }

foreach ($j in $jobs) {
    $which = $j[0]; $tag = $j[1]; $ck = $j[2]; $cdir = $j[3]
    foreach ($corr in $corrList) {
        foreach ($rho in $rhoList) {
            $rtag = ($rho -replace "\.", "")
            foreach ($seed in $seedList) {
                $out = "results/natural/nat_${tag}_${corr}_rho${rtag}_s${seed}.npz"
                if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
                Run "natural $tag / $corr / rho=$rho / seed $seed" "python -m experiments.exp_natural --which $which --ckpt $ck --cdir $cdir --corruption $corr --adapters '$Adapters' --rho $rho --bs $Bs --steps $Steps --eval_every $EvalEvery --seed $seed --out $out"
            }
        }
    }
}

Run "natural-feedback summary" "python -m experiments.summarize_natural --glob 'results/natural/nat_*.npz' --out results/natural.csv"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Natural-feedback table + verdict: results\natural.csv  (and the VERDICT lines above)"
Write-Host "Full log: results\natural_$ts.log"
Stop-Transcript | Out-Null
