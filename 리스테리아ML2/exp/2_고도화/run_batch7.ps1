# 2nd-round batch 7: m (M1, A3_M1) -> c (C4_base, C4_hist) -> q (Q1~Q4 growth range) -> n (A3_N1, A3_N2, A3_N1N2, A3_N0) -> tables
# n is last because A3_N0 needs the most memory (private peak ~6.4GB); if it stops, earlier stages are kept and a rerun resumes.
# writes logs\batch7_exit.txt with per-step exit codes and times
$here = $PSScriptRoot
$ps1 = Join-Path $here 'run_exp2.ps1'
$out = Join-Path $here 'logs\batch7_exit.txt'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Confirm:$false }
$steps = @('m', 'c', 'q', 'n', 'tables')
$lines = @()
foreach ($st in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    & $ps1 -Stage $st
    $code = $LASTEXITCODE
    $lines += ("{0} exit={1} start={2} end={3}" -f $st, $code, $t0, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    if ($code -ne 0 -and $st -ne 'n') { break }
}
$lines += ("batch7 done " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
