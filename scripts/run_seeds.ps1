# Multi-seed overnight run: probe (tent, plst) + rescue across seeds, then
# aggregate into mean +/- std and render the final phase figures.
# One command:  powershell -ExecutionPolicy Bypass -File scripts\run_seeds.ps1
#   add -MaxGpu to take the whole card (default is courtesy: won't disturb other jobs).
# Resumable: already-finished seeds are skipped, so re-run anytime to continue.
param(
    [string]$Data = "D:/data",
    [string]$Seeds = "0,1,2,3,4",
    [string]$Adapters = "tent,plst",
    [string]$Eps = "0,0.2,0.4,0.6,0.8,1.0",
    [string]$Bs = "4,8,16,32,64,128",
    [int]$Steps = 1500,
    [int]$EvalEvery = 50,
    [int]$RescueBs = 16,          # the collapse cell we validate PATTA on
    [double]$RescueEps = 1.0,
    [double]$RescueLr = 0.001,
    [int]$RescueSteps = 1500,
    [string]$MemFraction = "0.35",
    [int]$MinFreeMB = 3000,
    [switch]$MaxGpu
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

# ---------- GPU courtesy (default) vs -MaxGpu ----------
if ($MaxGpu) {
    $env:PERFTTA_GPU_MEM_FRACTION = "0.9"
    $env:PERFTTA_GPU_MIN_FREE_MB  = "500"
    try { (Get-Process -Id $PID).PriorityClass = 'Normal' } catch {}
    Write-Host "MAX-GPU mode: taking the card."
} else {
    $env:PERFTTA_GPU_MEM_FRACTION = "$MemFraction"
    $env:PERFTTA_GPU_MIN_FREE_MB  = "$MinFreeMB"
    try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}
    Write-Host "Courtesy mode: cap=$MemFraction  min-free=${MinFreeMB}MB  priority=BelowNormal"
}
$env:PERFTTA_GPU_WAIT_TIMEOUT_S = "10800"
$env:PERFTTA_GPU_POLL_S         = "30"
$env:PYTORCH_CUDA_ALLOC_CONF    = "max_split_size_mb:128"

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "results/seeds" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\seeds_$ts.log" -Append | Out-Null
$startTime = Get-Date

function Section($m) { Write-Host "`n==================== $m ====================" }
function Run($desc, $cmd) {
    Section $desc
    Write-Host ">> $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

function Ensure($url, $tar, $pooldir, $expected) {
    if (Test-Path "$pooldir/labels.npy") { Write-Host "pool present: $pooldir"; return $true }
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        if ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected)) { break }
        Write-Host "downloading -> $tar  (attempt $attempt/5, resumes if partial)"
        curl.exe -L --fail --retry 10 --retry-delay 15 --retry-all-errors -C - -o $tar $url
        Start-Sleep -Seconds 5
    }
    if (-not ((Test-Path $tar) -and ((Get-Item $tar).Length -ge $expected))) {
        Write-Host "[warn] could not fully download $tar"; return $false
    }
    tar.exe -xf $tar -C (Split-Path $pooldir -Parent)
    return (Test-Path "$pooldir/labels.npy")
}

# ---------- CUDA hard gate ----------
Section "PyTorch / CUDA"
python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FATAL] CUDA not available - aborting (reinstall the cu128 wheel)."
    Stop-Transcript | Out-Null; exit 1
}

# ---------- data + source checkpoint (CIFAR-10) ----------
New-Item -ItemType Directory -Force -Path "$Data/cifar10c","$Data/checkpoints","$Data/cifar" | Out-Null
$c10  = "$Data/cifar10c/CIFAR-10-C"
$ck10 = "$Data/checkpoints/resnet18_cifar10.pt"
Section "CIFAR-10-C data -> $Data"
$have10 = Ensure "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1" "$Data/cifar10c/CIFAR-10-C.tar" $c10 2900000000
if (-not $have10) { Write-Host "[FATAL] CIFAR-10-C unavailable - aborting."; Stop-Transcript | Out-Null; exit 1 }
if (-not (Test-Path $ck10)) {
    Run "Train source CIFAR-10" "python -m experiments.train_source --which cifar10 --epochs 30 --bs 128 --out $ck10"
}

$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$adapterList = $Adapters.Split(",") | ForEach-Object { $_.Trim() }
$isDefaultGrid = ($Eps -eq "0,0.2,0.4,0.6,0.8,1.0") -and ($Bs -eq "4,8,16,32,64,128") -and ($Steps -eq 1500) -and ($EvalEvery -eq 50)

# reuse the seed-0 probes already computed (only when the grid matches)
$canon = @{ "tent" = "results/probe_tent_bacc.npz"; "plst" = "results/probe_plst_bacc.npz" }
if ($isDefaultGrid) {
    foreach ($ad in $adapterList) {
        $s0 = "results/seeds/probe_${ad}_s0.npz"
        if ((-not (Test-Path $s0)) -and $canon.ContainsKey($ad) -and (Test-Path $canon[$ad])) {
            Copy-Item $canon[$ad] $s0; Write-Host "reused seed-0 $ad probe -> $s0"
        }
    }
}

# ---------- probes: adapter x seed ----------
foreach ($ad in $adapterList) {
    foreach ($seed in $seedList) {
        $out = "results/seeds/probe_${ad}_s${seed}.npz"
        if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
        Run "probe $ad / seed $seed" "python -m experiments.exp_collapse_probe --which cifar10 --ckpt $ck10 --cdir $c10 --adapter $ad --optim adam --lr 0.001 --steps $Steps --eps '$Eps' --bs '$Bs' --eval_every $EvalEvery --seed $seed --out $out"
    }
}

# ---------- rescue: validate PATTA at the collapse cell, per seed ----------
foreach ($seed in $seedList) {
    $out = "results/seeds/rescue_s${seed}.npz"
    if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
    Run "rescue / seed $seed" "python -m experiments.exp_patta_rescue --which cifar10 --ckpt $ck10 --cdir $c10 --eps $RescueEps --lr $RescueLr --optim adam --bs $RescueBs --steps $RescueSteps --eval_every $EvalEvery --seed $seed --out $out"
}

# ---------- aggregate + final figures ----------
foreach ($ad in $adapterList) {
    Run "aggregate probe $ad" "python -m experiments.aggregate_seeds --kind probe --glob 'results/seeds/probe_${ad}_s*.npz' --out results/probe_${ad}_mean.npz"
}
Run "aggregate rescue" "python -m experiments.aggregate_seeds --kind rescue --glob 'results/seeds/rescue_s*.npz' --out results/rescue_mean.npz"

$meanNpz = ($adapterList | ForEach-Object { "results/probe_${_}_mean.npz" }) -join " "
$labels  = ($adapterList | ForEach-Object { $_.ToUpper() }) -join " "
Run "phase diagram (mean balanced acc)" "python -m experiments.plot_phase --npz $meanNpz --label $labels --metric bacc --out results/phase_bacc_mean.png"
Run "phase diagram (mean H(q))"          "python -m experiments.plot_phase --npz $meanNpz --label $labels --metric Hq   --out results/phase_Hq_mean.png"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Mean grids:   results\probe_tent_mean.npz, results\probe_plst_mean.npz"
Write-Host "Rescue stats: results\rescue_mean.npz / .csv  (TENT vs PATTA balanced acc, N seeds)"
Write-Host "Figures:      results\phase_bacc_mean.png, results\phase_Hq_mean.png"
Write-Host "Full log:     results\seeds_$ts.log"
Stop-Transcript | Out-Null
