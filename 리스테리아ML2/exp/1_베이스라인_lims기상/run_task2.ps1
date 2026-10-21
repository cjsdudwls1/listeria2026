# 과업 2 전체 실행 (python src\run.py --stage all = step1 -> step2 -> tables)
# 별도 PowerShell 창에서 실행:
#   powershell -NoExit -ExecutionPolicy Bypass -File "C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\1_베이스라인_lims기상\run_task2.ps1"
# - 다시 실행하면 결과가 완전한 실행은 건너뛰고 이어서 한다.
# - 콘솔 출력은 logs\console.log (UTF-8), 진행 기록은 logs\progress.log 에 남는다.
# - 오류가 나면 숨기지 않고 멈춘다(종료 코드 1).

$here = $PSScriptRoot
$py = 'C:\Users\NT751\Desktop\천영진\리스테리아ML\.venv-exp2\Scripts\python.exe'
Set-Location $here
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
New-Item -ItemType Directory -Force -Path (Join-Path $here 'logs') | Out-Null
$consoleLog = Join-Path $here 'logs\console.log'

Write-Host ("[{0}] 과업 2 시작 (python: {1})" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $py)
& $py -u (Join-Path $here 'src\run.py') --stage all --console-log $consoleLog
$code = $LASTEXITCODE
if ($code -eq 0) {
    Write-Host ("[{0}] 완료. 결과표: tables\task2_결과표.xlsx" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
} else {
    Write-Host ("[{0}] 오류로 멈춤 (종료 코드 {1}). logs\progress.log, logs\console.log 확인. 고친 뒤 다시 실행하면 이어서 함." -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $code)
}
exit $code