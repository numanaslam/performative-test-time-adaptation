# SCALE run: does the performative collapse / hidden-metric / PATTA-recovery story hold
# at ImageNet scale (1000 classes, 224px, ResNet-50)?  1000 classes make the hidden
# metric extreme (chance = 0.001).
#   1) scripts\setup_imagenet_c.ps1 -Categories "noise,digital"   (one-time, large download)
#   2) powershell -ExecutionPolicy Bypass -File scripts\run_scale.ps1 -MaxGpu
# Resumable (skips finished runs).
param(
    [string]$Data = "D:/data",
    [string]$Arch = "resnet50",
    [string]$Seeds = "0,1",
    [string]$Corruptions = "gaussian_noise,contrast",
    [string]$Severities = "3,5",      # sev5 = floor (hidden-metric only); sev3 = healthy start (rescue test); add 2 to widen
    [string]$Epsilons = "0,1",        # eps=0 reference vs eps=1 performative -> ISOLATES the feedback effect at scale
    [string]$Adapters = "source,tent,balance,patta,patta_sev",
    [int]$Bs = 16,
    [int]$Steps = 1000,
    [int]$EvalEvery = 200,
    [int]$EvalBs = 1000,          # 1 sample/class at 1000 classes; chunked forward (eval_chunk=128) avoids OOM
    [switch]$MaxGpu
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
if ($MaxGpu) {
    $env:PERFTTA_GPU_MEM_FRACTION = "0.9"; $env:PERFTTA_GPU_MIN_FREE_MB = "500"
    try { (Get-Process -Id $PID).PriorityClass = 'Normal' } catch {}; Write-Host "MAX-GPU mode."
} else {
    $env:PERFTTA_GPU_MEM_FRACTION = "0.5"; $env:PERFTTA_GPU_MIN_FREE_MB = "2000"
    try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}; Write-Host "Courtesy mode."
}
$env:PERFTTA_GPU_WAIT_TIMEOUT_S = "10800"; $env:PERFTTA_GPU_POLL_S = "30"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"   # ResNet-50/224px: reduce fragmentation (per the OOM hint)

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "results/scale" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "results\scale_$ts.log" -Append | Out-Null
$startTime = Get-Date

function Section($m) { Write-Host "`n==================== $m ====================" }
function Run($desc, $cmd) {
    Section $desc; Write-Host ">> $cmd"; Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] step failed (exit $LASTEXITCODE): $desc" }
}

Section "PyTorch / CUDA"
python -c "import torch,sys; print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] CUDA not available."; Stop-Transcript | Out-Null; exit 1 }

$cdir = "$Data/imagenet-c"
$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$corrList = $Corruptions.Split(",") | ForEach-Object { $_.Trim() }
$sevList  = $Severities.Split(",") | ForEach-Object { [int]$_.Trim() }
$epsList  = $Epsilons.Split(",") | ForEach-Object { [double]$_.Trim() }

foreach ($corr in $corrList) {
    foreach ($sev in $sevList) {
        $sdir = Join-Path $cdir (Join-Path $corr "$sev")
        if (-not (Test-Path $sdir)) { Write-Host "[note] $sdir missing (extract this severity) - skipping $corr sev$sev"; continue }
        $nc = (Get-ChildItem $sdir -Directory).Count
        if ($nc -ne 1000) { Write-Host "[WARN] $corr sev$sev has $nc class folders, not 1000 - needs full 1000. Skipping."; continue }
        foreach ($epsv in $epsList) {
            $etag = ($epsv.ToString() -replace "\.", "")
            foreach ($seed in $seedList) {
                $out = "results/scale/scale_in_${corr}_sev${sev}_eps${etag}_bs${Bs}_s${seed}.npz"
                if (Test-Path $out) { Write-Host "skip (exists): $out"; continue }
                Run "scale imagenet / $corr / sev=$sev / eps=$epsv / bs=$Bs / seed $seed" "python -m experiments.exp_scale --cdir $cdir --arch $Arch --corruption $corr --severity $sev --adapters '$Adapters' --eps $epsv --bs $Bs --steps $Steps --eval_every $EvalEvery --eval_bs $EvalBs --seed $seed --out $out"
            }
        }
    }
}

Run "scale sweep summary" "python -m experiments.summarize_scale --glob 'results/scale/scale_in_*_sev*.npz' --out results/scale.csv"

$mins = [int]((Get-Date) - $startTime).TotalMinutes
Section "ALL DONE in ~$mins min"
Write-Host "Scale table: results\scale.csv  (per corruption: each adapter balanced vs stream acc at 1000 classes)"
Write-Host "Full log: results\scale_$ts.log"
Stop-Transcript | Out-Null
