# 2nd-round batch 3: D2 (SMOTE, within-province neighbors + base-row categories) then tables
# writes logs\batch3_exit.txt with per-step exit codes and times
$here = $PSScriptRoot
$ps1 = Join-Path $here 'run_exp2.ps1'
$out = Join-Path $here 'logs\batch3_exit.txt'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Confirm:$false }
$steps = @(
    @{ Stage = 'd';      Only = 'D2' },
    @{ Stage = 'tables'; Only = '' }
)
$lines = @()
foreach ($s in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ($s.Only) { & $ps1 -Stage $s.Stage -Only $s.Only } else { & $ps1 -Stage $s.Stage }
    $code = $LASTEXITCODE
    $lines += ("{0} only={1} exit={2} start={3} end={4}" -f $s.Stage, $s.Only, $code, $t0, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    if ($code -ne 0) { break }
}
$lines += ("batch3 done " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
