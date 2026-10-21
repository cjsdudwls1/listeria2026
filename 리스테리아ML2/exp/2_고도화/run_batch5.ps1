# 2nd-round batch 5: extra seeds (43, 44) for fallback combos A3_F4, A3_G1G2G3, then tables
# writes logs\batch5_exit.txt with per-step exit codes and times
$here = $PSScriptRoot
$ps1 = Join-Path $here 'run_exp2.ps1'
$out = Join-Path $here 'logs\batch5_exit.txt'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Confirm:$false }
$steps = @('combo', 'tables')
$lines = @()
foreach ($st in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    & $ps1 -Stage $st
    $code = $LASTEXITCODE
    $lines += ("{0} exit={1} start={2} end={3}" -f $st, $code, $t0, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    if ($code -ne 0) { break }
}
$lines += ("batch5 done " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
