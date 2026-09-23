$ErrorActionPreference = "Stop"
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + $env:PATH
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

& "C:\msys64\ucrt64\bin\g++.exe" -std=c++17 -O2 -fno-use-linker-plugin `
  -I "$root\include" -I "$root\third_party\eigen-3.4.0" `
  "$root\tests\test_cpp.cpp" -o "$root\tests\test_cpp.exe"

if ($LASTEXITCODE -ne 0) { Write-Host "COMPILE FAILED"; exit $LASTEXITCODE }
& "$root\tests\test_cpp.exe"
exit $LASTEXITCODE
