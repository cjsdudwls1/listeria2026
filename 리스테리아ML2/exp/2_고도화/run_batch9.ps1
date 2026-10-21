# 2nd-round batch 9: p (P4, P4p HACCP; finished P1~P3 runs are skipped) -> tables
# writes logs\batch9_exit.txt with per-step exit codes and times
$here = $PSScriptRoot
$ps1 = Join-Path $here 'run_exp2.ps1'
$out = Join-Path $here 'logs\batch9_exit.txt'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Confirm:$false }
$steps = @('p', 'tables')
$lines = @()
foreach ($st in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    & $ps1 -Stage $st
    $code = $LASTEXITCODE
    $lines += ("{0} exit={1} start={2} end={3}" -f $st, $code, $t0, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    if ($code -ne 0) { break }
}
$lines += ("batch9 done " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
