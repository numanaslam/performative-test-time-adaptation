#!/usr/bin/env bash
# Fallback for WSL / Git-Bash. Set DATA to your data root (default D:/data).
set -euo pipefail
DATA="${DATA:-D:/data}"
mkdir -p "$DATA/cifar10c" "$DATA/cifar100c" "$DATA/checkpoints" "$DATA/cifar"

dl() { url="$1"; out="$2"; [ -f "$out" ] && echo "exists: $out" || curl -L --fail -C - -o "$out" "$url"; }
dl "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"  "$DATA/cifar10c/CIFAR-10-C.tar"
dl "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1" "$DATA/cifar100c/CIFAR-100-C.tar"

echo "extracting..."
tar -xf "$DATA/cifar10c/CIFAR-10-C.tar"   -C "$DATA/cifar10c"
tar -xf "$DATA/cifar100c/CIFAR-100-C.tar" -C "$DATA/cifar100c"
echo "done. pools at $DATA/cifar10c/CIFAR-10-C and $DATA/cifar100c/CIFAR-100-C"
