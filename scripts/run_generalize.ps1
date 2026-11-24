# Collapse-regime GENERALIZATION: does the harm + PATTA fix hold across
# corruptions and on CIFAR-100? Runs the rescue at the collapse cell for every
# (dataset x corruption x seed), plus a CIFAR-100 phase probe, then summarizes.
# One command:  powershell -ExecutionPolicy Bypass -File scripts\run_generalize.ps1
#   add -MaxGpu to take the whole card. Resumable (skips finished runs).
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2",
    [string]$Corruptions = "gaussian_noise,shot_noise,fog,contrast",
    [string]$ProbeSeeds = "0",                 # CIFAR-100 phase probe (expensive) -> fewer seeds
    [int]$RescueBs = 16,                        # the collapse cell
    [double]$RescueEps = 1.0,
    [double]$RescueLr = 0.001,
    [int]$RescueSteps = 1500,
    [int]$EvalEvery = 50,
    [switch]$SkipC100Probe,
    [string]$MemFraction = "0.35",
    [int]$MinFreeMB = 3000,
    [switch]$MaxGpu
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
if ($MaxGpu) {
    $env:PERFTTA_GPU_MEM_FRACTION = "0.9"; $env:PERFTTA_GPU_MIN_FREE_MB = "500"
    try { (Get-Process -Id $PID).PriorityClass = 'Normal' } catch {}
    Write-Host "MAX-GPU mode."
} else {
    $env:PERFTTA_GPU_MEM_FRACTION = "$MemFraction"; $env:PERFTTA_GPU_MIN_FREE_MB = "$MinFreeMB"
    try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}
    Write-Host "Courtesy mode: cap=$MemFraction min-free=${MinFreeMB}MB priority=BelowNormal"
}
$env:PERFTTA_GPU_WAIT_TIMEOUT_S = "10800"; $env:PERFTTA_GPU_POLL_S = "30"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "results/gen" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\generalize_$ts.log" -Append | Out-Null
$startTime = Get-Date

function Section($m) { Write-Host "`n==================== $m ====================" }
function Run($desc, $cmd) {
    Section $desc; Write-Host ">> $cmd"; Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}
function Ensure($url, $tar, $pooldir, $expected) {
    if (Test-Path "$pooldir/labels.npy") { Write-Host "pool present: $pooldir"; return $true }
    for ($a = 1; $a -le 5; $a++) {
        if ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected)) { break }
        curl.exe -L --fail --retry 10 --retry-delay 15 --retry-all-errors -C - -o $tar $url
        Start-Sleep -Seconds 5
    }
    if (-not ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected))) { return $false }
    tar.exe -xf $tar -C (Split-Path $pooldir -Parent)
    return (Test-Path "$pooldir/labels.npy")
}

Section "PyTorch / CUDA"
python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] CUDA not available."; Stop-Transcript | Out-Null; exit 1 }

# ---------- data + checkpoints ----------
New-Item -ItemType Directory -Force -Path "$Data/cifar10c","$Data/cifar100c","$Data/checkpoints","$Data/cifar" | Out-Null
$c10 = "$Data/cifar10c/CIFAR-10-C";  $ck10 = "$Data/checkpoints/resnet18_cifar10.pt"
$c100 = "$Data/cifar100c/CIFAR-100-C"; $ck100 = "$Data/checkpoints/resnet18_cifar100.pt"

$have10 = Ensure "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1" "$Data/cifar10c/CIFAR-10-C.tar" $c10 2900000000
if (-not $have10) { Write-Host "[FATAL] CIFAR-10-C unavailable."; Stop-Transcript | Out-Null; exit 1 }
if (-not (Test-Path $ck10)) { Run "Train source CIFAR-10" "python -m experiments.train_source --which cifar10 --epochs 30 --bs 128 --out $ck10" }

$have100 = Ensure "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1" "$Data/cifar100c/CIFAR-100-C.tar" $c100 2900000000
if ($have100 -and -not (Test-Path $ck100)) { Run "Train source CIFAR-100" "python -m experiments.train_source --which cifar100 --epochs 30 --bs 128 --out $ck100" }
if (-not $have100) { Write-Host "[note] CIFAR-100-C unavailable - CIFAR-100 runs skipped (re-run later to resume)." }

$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$corrList = $Corruptions.Split(",") | ForEach-Object { $_.Trim() }
$probeSeedList = $ProbeSeeds.Split(",") | ForEach-Object { [int]$_.Trim() }

# datasets to cover: (which, tag, ckpt, cdir)
$sets = @()
$sets += ,@("cifar10", "c10", $ck10, $c10)
if ($have100) { $sets += ,@("cifar100", "c100", $ck100, $c100) }

# ---------- rescue at the collapse cell: dataset x corruption x seed ----------
foreach ($s in $sets) {
    $which = $s[0]; $tag = $s[1]; $ck = $s[2]; $cdir = $s[3]
    foreach ($corr in $corrList) {
        foreach ($seed in $seedList) {
            $out = "results/gen/rescue_${tag}_${corr}_s${seed}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "rescue $tag / $corr / seed $seed" "python -m experiments.exp_patta_rescue --which $which --ckpt $ck --cdir $cdir --corruption $corr --eps $RescueEps --lr $RescueLr --optim adam --bs $RescueBs --steps $RescueSteps --eval_every $EvalEvery --seed $seed --out $out"
        }
    }
}

# ---------- CIFAR-100 phase probe (bifurcation on 100 classes) ----------
if ($have100 -and -not $SkipC100Probe) {
    foreach ($ad in @("tent", "plst")) {
        foreach ($seed in $probeSeedList) {
            $out = "results/gen/probe_c100_${ad}_s${seed}.npz"
            if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
            Run "probe cifar100 $ad / seed $seed" "python -m experiments.exp_collapse_probe --which cifar100 --ckpt $ck100 --cdir $c100 --corruption gaussian_noise --adapter $ad --optim adam --lr 0.001 --steps 1500 --eps '0,0.2,0.4,0.6,0.8,1.0' --bs '4,8,16,32,64,128' --eval_every $EvalEvery --seed $seed --out $out"
        }
    }
}

# ---------- summarize + figures ----------
Run "generalization summary" "python -m experiments.summarize_generalization --glob 'results/gen/rescue_*.npz' --out results/generalization.csv"
if ($have100 -and -not $SkipC100Probe) {
    Run "aggregate cifar100 probe tent" "python -m experiments.aggregate_seeds --kind probe --glob 'results/gen/probe_c100_tent_s*.npz' --which cifar100 --out results/probe_c100_tent_mean.npz"
    Run "aggregate cifar100 probe plst" "python -m experiments.aggregate_seeds --kind probe --glob 'results/gen/probe_c100_plst_s*.npz' --which cifar100 --out results/probe_c100_plst_mean.npz"
    Run "cifar100 phase (balanced acc)" "python -m experiments.plot_phase --npz results/probe_c100_tent_mean.npz results/probe_c100_plst_mean.npz --label TENT PLST --which cifar100 --metric bacc --out results/phase_c100_bacc_mean.png"
    Run "cifar100 phase (H(q))"          "python -m experiments.plot_phase --npz results/probe_c100_tent_mean.npz results/probe_c100_plst_mean.npz --label TENT PLST --which cifar100 --metric Hq --out results/phase_c100_Hq_mean.png"
}

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Generalization table: results\generalization.csv"
Write-Host "CIFAR-100 figures:    results\phase_c100_bacc_mean.png, results\phase_c100_Hq_mean.png"
Write-Host "Full log:             results\generalize_$ts.log"
Stop-Transcript | Out-Null
