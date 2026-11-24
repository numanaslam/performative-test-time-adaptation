# Overnight end-to-end run: setup -> data -> train -> all experiments.
# One command:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# - Does NOT disturb other GPU jobs: waits for free VRAM, caps its share, small
#   batch, below-normal priority.
# - Network-robust: downloads retry + resume on blips.
# - Graceful: CIFAR-10 pipeline runs first; CIFAR-100 is skipped (not fatal) if
#   its data can't be fetched. Resumable: re-run anytime to continue.
param(
    [string]$Data = "D:/data",
    [int]$Bs = 64,                       # small batch -> ~1 GB footprint
    [string]$MemFraction = "0.35",       # cap THIS run to ~35% of the 12 GB card
    [int]$MinFreeMB = 3000               # wait for >=3 GB free before each phase
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

# ---------- GPU courtesy: do not disturb already-running jobs ----------
$env:PERFTTA_GPU_MEM_FRACTION   = "$MemFraction"
$env:PERFTTA_GPU_MIN_FREE_MB    = "$MinFreeMB"
$env:PERFTTA_GPU_WAIT_TIMEOUT_S = "10800"
$env:PERFTTA_GPU_POLL_S         = "30"
$env:PYTORCH_CUDA_ALLOC_CONF    = "max_split_size_mb:128"
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "results" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\run_$ts.log" -Append | Out-Null
$startTime = Get-Date
Write-Host "GPU courtesy: cap=$MemFraction  min-free=${MinFreeMB}MB  batch=$Bs  priority=BelowNormal"

function Section($m) { Write-Host "`n==================== $m ====================" }
function Run($desc, $cmd) {
    Section $desc
    Write-Host ">> $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

# Download + extract one dataset. Returns $true on success. Retries & resumes.
function Ensure($url, $tar, $pooldir, $expected) {
    if (Test-Path "$pooldir/labels.npy") { Write-Host "pool present: $pooldir"; return $true }
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        if ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected)) { break }
        Write-Host "downloading -> $tar  (attempt $attempt/5, resumes if partial)"
        curl.exe -L --fail --retry 10 --retry-delay 15 --retry-all-errors -C - -o $tar $url
        if ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected)) { break }
        Write-Host "[warn] attempt $attempt incomplete (curl exit $LASTEXITCODE); retrying in 20s..."
        Start-Sleep -Seconds 20
    }
    if (-not ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected))) {
        Write-Host "[warn] could not fully download $tar"
        return $false
    }
    Write-Host "extracting -> $pooldir"
    tar.exe -xf $tar -C (Split-Path $pooldir -Parent)
    return (Test-Path "$pooldir/labels.npy")
}

# ---------- 0. PyTorch + CUDA (hard gate) ----------
Section "PyTorch / CUDA"
python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing torch (cu128 for Blackwell/5070)..."
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
}
pip install -r requirements.txt
python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY')); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FATAL] CUDA not available - aborting. Reinstall the cu128 wheel (README section 1)."
    Stop-Transcript | Out-Null; exit 1
}

# ---------- 1. Smoke test ----------
Run "Smoke test (no data needed)" "python -m experiments.smoke_test"

New-Item -ItemType Directory -Force -Path "$Data/cifar10c","$Data/cifar100c","$Data/checkpoints","$Data/cifar" | Out-Null

# ================= CIFAR-10 (primary: run this FIRST) =================
Section "CIFAR-10-C data -> $Data"
$have10 = Ensure "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1" "$Data/cifar10c/CIFAR-10-C.tar" "$Data/cifar10c/CIFAR-10-C" 2900000000
if (-not $have10) { Write-Host "[FATAL] CIFAR-10-C (primary dataset) unavailable - aborting."; Stop-Transcript | Out-Null; exit 1 }

$ck10 = "$Data/checkpoints/resnet18_cifar10.pt"
$c10  = "$Data/cifar10c/CIFAR-10-C"
if (Test-Path $ck10) { Write-Host "cifar10 checkpoint exists, skipping training" }
else { Run "Train source CIFAR-10" "python -m experiments.train_source --which cifar10 --epochs 30 --bs $Bs --out $ck10" }

foreach ($corr in @("gaussian_noise", "fog", "contrast")) {
    Run "E1 bifurcation cifar10 / $corr" "python -m experiments.exp_bifurcation --which cifar10 --ckpt $ck10 --cdir $c10 --corruption $corr --severity 5 --steps 250 --bs $Bs --out results/bif_c10_$corr.npz"
}
Run "E2 PATTA rescue cifar10"          "python -m experiments.exp_patta_rescue --which cifar10 --ckpt $ck10 --cdir $c10 --eps 0.8 --lr 0.0025 --steps 400 --bs $Bs --out results/rescue_c10.npz"
Run "E3 online cifar10 (all 15 corruptions)" "python -m experiments.exp_online --which cifar10 --ckpt $ck10 --cdir $c10 --corruptions all --eps 0.6 --steps 250 --bs $Bs --out results/online_c10.csv"

# ================= CIFAR-100 (secondary: skip if data won't fetch) =================
Section "CIFAR-100-C data -> $Data"
$have100 = Ensure "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1" "$Data/cifar100c/CIFAR-100-C.tar" "$Data/cifar100c/CIFAR-100-C" 2900000000
if (-not $have100) {
    Write-Host "[note] CIFAR-100-C unavailable - skipping CIFAR-100 experiments. Re-run this script later to resume its download and finish them."
} else {
    $ck100 = "$Data/checkpoints/resnet18_cifar100.pt"
    $c100  = "$Data/cifar100c/CIFAR-100-C"
    if (Test-Path $ck100) { Write-Host "cifar100 checkpoint exists, skipping training" }
    else { Run "Train source CIFAR-100" "python -m experiments.train_source --which cifar100 --epochs 30 --bs $Bs --out $ck100" }

    Run "E1 bifurcation cifar100 / gaussian_noise" "python -m experiments.exp_bifurcation --which cifar100 --ckpt $ck100 --cdir $c100 --corruption gaussian_noise --severity 5 --steps 250 --bs $Bs --out results/bif_c100_gaussian_noise.npz"
    Run "E2 PATTA rescue cifar100"                 "python -m experiments.exp_patta_rescue --which cifar100 --ckpt $ck100 --cdir $c100 --eps 0.8 --lr 0.0025 --steps 400 --bs $Bs --out results/rescue_c100.npz"
    Run "E3 online cifar100"                        "python -m experiments.exp_online --which cifar100 --ckpt $ck100 --cdir $c100 --corruptions gaussian_noise,fog,contrast --eps 0.6 --steps 250 --bs $Bs --out results/online_c100.csv"
}

# ---------- done ----------
$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Results: results\   (bif_*.npz/.png, rescue_*.npz, online_*.csv)"
Write-Host "Full log: results\run_$ts.log"
Stop-Transcript | Out-Null
