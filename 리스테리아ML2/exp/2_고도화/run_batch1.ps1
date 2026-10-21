# 2차 실험 1차분: 과거 이력 변수 결정과 무관한 실험만 차례로 실행
# (E00 -> 1차 가중 지표 -> D1 클래스 가중치 -> H10 -> H30 -> 딥러닝 4종)
# 과거 이력 변수(A·B·C 이력판·D3 WOE)와 D2(SMOTE)는 사용자 결정 뒤에 따로 실행
# 끝나면 logs\batch1_exit.txt 에 단계별 종료 코드와 시각을 씀
$here = $PSScriptRoot
$ps1 = Join-Path $here 'run_exp2.ps1'
$out = Join-Path $here 'logs\batch1_exit.txt'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Confirm:$false }
$steps = @(
    @{ Stage = 'e00';   Only = '' },
    @{ Stage = 'prev1'; Only = '' },
    @{ Stage = 'd';     Only = 'D1' },
    @{ Stage = 'h10';   Only = '' },
    @{ Stage = 'hbig';  Only = '' },
    @{ Stage = 'dl';    Only = '' }
)
$lines = @()
foreach ($s in $steps) {
    $t0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ($s.Only) { & $ps1 -Stage $s.Stage -Only $s.Only } else { & $ps1 -Stage $s.Stage }
    $code = $LASTEXITCODE
    $lines += ("{0} only={1} exit={2} start={3} end={4}" -f $s.Stage, $s.Only, $code, $t0, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    if ($code -ne 0) { break }
}
$lines += ("batch1 done " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
