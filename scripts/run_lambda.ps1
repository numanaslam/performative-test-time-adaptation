# LAMBDA-SCHEDULE study: can a smarter marginal-entropy SCHEDULE reclaim the
# CIFAR-100 near-total-collapse regime (where plain PATTA under-corrected and the
# fixed `balance` heuristic won) without losing PATTA's CIFAR-10 edge or benign
# inertness?  All variants share lam0=1.0 (same MAX strength as balance); only the
# schedule shape differs:
#   patta  (u),  patta_steep (u^.5),  patta_floor (max .1,u),  patta_sev (max .1,u^.5)
# One command:  powershell -ExecutionPolicy Bypass -File scripts\run_lambda.ps1 -MaxGpu
# Resumable (skips finished runs).
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",
    [string]$Corruptions = "gaussian_noise,contrast",
    [string]$Adapters = "tent,balance,scalar,patta,patta_steep,patta_floor,patta_sev",
    [double]$Eps = 1.0,
    [int]$Steps = 1500,
    [int]$EvalEvery = 50,
    [int]$InertBs = 128,          # benign no-collapse cell for the inertness check
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
New-Item -ItemType Directory -Force -Path "results/lambda","results/lambda_inert" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\lambda_$ts.log" -Append | Out-Null
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

# MAIN grid (eps=$Eps): c10 bs16 (PATTA already won -> regression check),
#   c100 bs16 + bs32 (PATTA lost to balance -> reclaim target).
$jobs = @()
$jobs += ,@("cifar10", "c10", $ck10, $c10, 16)
if ($have100) {
    $jobs += ,@("cifar100", "c100", $ck100, $c100, 16)
    $jobs += ,@("cifar100", "c100", $ck100, $c100, 32)
}

foreach ($j in $jobs) {
    $which = $j[0]; $tag = $j[1]; $ck = $j[2]; $cdir = $j[3]; $b = $j[4]
    foreach ($corr in $corrList) {
        foreach ($seed in $seedList) {
            $out = "results/lambda/lam_${tag}_${corr}_bs${b}_s${seed}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "lambda $tag / $corr / bs=$b / seed $seed" "python -m experiments.exp_baselines --which $which --ckpt $ck --cdir $cdir --corruption $corr --adapters '$Adapters' --eps $Eps --bs $b --steps $Steps --eval_every $EvalEvery --seed $seed --out $out"
        }
    }
}

# INERTNESS check: benign no-collapse cell (c10, large batch, eps=0).
# The schedule variants must stay ~= tent here; the floor variants' benign tax is read off directly.
$inertAdapters = "tent,patta,patta_steep,patta_floor,patta_sev"
foreach ($corr in $corrList) {
    foreach ($seed in $seedList) {
        $out = "results/lambda_inert/inert_c10_${corr}_bs${InertBs}_eps0_s${seed}.npz"
        if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
        Run "inert c10 / $corr / bs=$InertBs / eps0 / seed $seed" "python -m experiments.exp_baselines --which cifar10 --ckpt $ck10 --cdir $c10 --corruption $corr --adapters '$inertAdapters' --eps 0.0 --bs $InertBs --steps $Steps --eval_every $EvalEvery --seed $seed --out $out"
    }
}

Run "lambda-schedule summary" "python -m experiments.summarize_lambda --glob 'results/lambda/lam_*.npz' --inert_glob 'results/lambda_inert/*.npz' --out results/lambda.csv"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Schedule table + verdict: results\lambda.csv  (and the RECOMMENDATION line above)"
Write-Host "Full log: results\lambda_$ts.log"
Stop-Transcript | Out-Null
