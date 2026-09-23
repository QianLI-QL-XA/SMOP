#======================================================================
# build_wsl.ps1 -- build & run the smop level-set BMOP test inside WSL
#
# Since Windows Application Control (Smart App Control) blocks freshly
# built exes, this script builds and runs inside WSL (Ubuntu), where no
# such policy exists.  Requirements (one-time):
#   - WSL with Ubuntu installed (wsl -l -v shows Ubuntu, Running, 2)
#   - inside Ubuntu: apt-get install -y g++ libeigen3-dev
#
# Usage in VS Code terminal (or any PowerShell):
#   powershell -ExecutionPolicy Bypass -File build_wsl.ps1 [dataName ...]
# Default data names: mpg_scale_expanded7 pyrim_scaled_expanded5 ...
#======================================================================
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\qianl\OneDrive\codes\SMOP\smop'
$names = if ($args.Count -gt 0) { ($args -join ' ') } else { 'mpg_scale_expanded7' }

$cmd = @"
set -e
mkdir -p /root/smop-build
ln -sfn '/mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/tests' /root/tests
g++ -std=c++17 -O3 -march=native -fopenmp \
  -I '/mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/include' \
  -I /usr/include/eigen3 \
  '/mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/examples/example_ucidata.cpp' \
  -o /root/smop-build/example_ucidata
cd /root/smop-build
SMOP_VERBOSE=2 ./example_ucidata $names
"@
Write-Host "[wsl] building & running: $names"
wsl -d Ubuntu -u root -- bash -lc $cmd
