# 2차(고도화) 실험 전체 실행 (python src\run.py --stage all)
# 별도 PowerShell 창에서 실행:
#   powershell -NoExit -ExecutionPolicy Bypass -File "C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\2_고도화\run_exp2.ps1"
#   단계만: ... -File "...\run_exp2.ps1" -Stage a      시험 실행: ... -Stage all -Smoke
# - 단계(-Stage): all, e00, prev1, a_k, a, b, c, d, h10, hbig, h60, dl, f, g, combo, m, n, q, p, hx, ae, xs, stk, tables
#   all 순서 = e00 -> prev1 -> a_k -> a -> b -> c -> d -> h10 -> hbig -> h60 -> dl -> f -> g -> combo -> m -> n -> q -> p -> hx -> ae -> xs -> stk -> tables (끝난 실험은 건너뜀)
#   f = F1~F4 외부 자료(1차 보고서 표 9 ①), g = G1~G3 생장식 변수(표 9 ②), combo = A3 + (검증에서 E00 보다 높았던 F·G 묶음)
#   (사용자 승인 2026-10-19 "싹다 ㄱ" 2차 추가분. -Stage all 로 실행하면 끝난 실험은 건너뛰고 f -> g -> combo -> tables 만 실제로 돎)
#   m = M1(E00 + 이동평균 차감 기상 57칸)·A3_M1(A3 + 같은 57칸) 시드 42·43·44 (1차 보고서 표 9 ② 이동평균 차감, 사용자 승인 2026-10-19)
#   c 에 C4_base·C4_hist 추가(C4 = 학습 구간 검출 1건 이상 품목 26개 전부 다운사이징, 사용자 지시 2026-10-19) - 끝난 C1~C3 는 건너뛰고 C4 만 돎
#   n = A3_N1(A3 + 과거 뭉치 이력 84)·A3_N2(A3 + 그날 뭉치 멀티핫 21)·A3_N1N2·A3_N0(A3 + 제품명 TF-IDF 500) 시드 42·43·44
#       (제품명 변수 - 교수님 계획서 4·5절, 사용자 승인 2026-10-19, 계산 = src\e2nb.py)
#   q = Q1(A3 + 일별 기상 범위 12)·Q2(A3 + 토양 pH 범위 3)·Q3(A3 + 식품 범위 4)·Q4(A3 + 셋 다 19) 시드 42·43·44
#       (리서치팀 생장 최적·한계 범위, 사용자 지시 - [질문] 세션 전달 2026-10-19, 파일 data\데이터셋\생장범위\, 계산 = src\e2fg.py rgwx·rgsoil·rgfood)
#   p = P1(A3 + 친환경 비율)·P2(A3 + 지하수 적합률)·P3(둘 다)·P1p·P2p(위약판)·P4(A3 + HACCP 지정 비율)·P4p(위약판) 시드 42·43·44 (외부 지표, 사용자 승인·확정 2026-10-19)
#   hx = A3_L1(A3 + 1년 log(1+검사 수) 3)·A3_U1(A3 + 1년 스무딩 검출률 사후 표준편차 3)·A3_L1U1(둘 다 6) 시드 42·43·44
#       (과거 이력 파생변수, 사용자 지시 2026-10-19, 계산 = src\e2hist.py 'logn'·'sd', 점검 = logs\selftest_hx.txt, 날짜 시계 확인표 = logs\selftest_hx_dateclock.csv)
#   ae = AE·VAE 이상 탐지(사용자 지시 2026-10-19, 계산 = src\e2ae.py, 점검 = logs\selftest_ae.txt) 시드 42·43·44, GPU 필요:
#       S_AEt·S_VAEt·S_AEs·S_VAEs(점수 단독, CatBoost 없음) -> A3_AEt·A3_VAEt·A3_AEs·A3_VAEs(A3 + 점수 1칸) -> A3_*_p(위약판: 예측일 T 의 달력 연도 안 섞기).
#       점수는 (모델, 시드)마다 S_* 에서 한 번 계산해 results\_ae_scores\ 에 저장 -> 같은 시드의 A3_*·A3_*_p 가 읽어 씀(다시 실행해도 메타가 같으면 읽음)
#   xs = 작은 실험 3개(사용자 지시 2026-10-19, 계획서 11절 3순위·6절 실험 A 빈칸) 시드 42·43·44, 기존 CatBoost 설정 그대로:
#       E00_R1(E00 + 1년 스무딩 검출률 3, k*)·E00_NW(E00 에서 기상 요약 190칸 뺌: 범주 4 + h)·A3_NW(A3 에서 기상 요약 뺌: 범주 4 + h + 이력 9) (점검 = logs\selftest_xs.txt)
#   stk = E08 스태킹(사용자 지시 2026-10-19, 계획서 7절 방식 C, 계산 = src\e2stk.py, 점검 = logs\selftest_stk.txt) 시드 42·43·44:
#       STK_E00(E00 + DL_lstmA + DL_trfA)·STK_A3(A3 + DL_lstmA + DL_trfA) 같은 시드 저장 예측의 logit 3칸 -> 로지스틱 메타 모델(재학습 없음).
#       메타 모델을 검증 구간으로 학습 -> 검증 점수는 메타 학습 구간(비교에는 시험 점수만, 표 순위에서 뺌). 특성 자료를 만들지 않음(E00·A3·DL 결과 필요)
#   권장 순서: -Stage m -> -Stage c -> -Stage n -> -Stage q -> -Stage p -> -Stage hx -> -Stage ae -> -Stage xs -> -Stage stk -> -Stage tables (또는 -Stage all: 끝난 실험은 건너뜀)
#   (hx·ae·xs·stk 는 2026-10-19 팀 공용 서버에서 실행 - 이 PC 메모리 부족. 결과 폴더만 results\ 로 가져옴)
# - 다시 실행하면 결과가 완전한 실행은 건너뛰고 이어서 한다.
# - 콘솔 출력은 logs\console.log (UTF-8), 진행 기록은 logs\progress.log.
# - 시작할 때 이전 logs\run_exit.txt 를 지우고 logs\run_start.txt 에 시작 시각을 쓴다.
#   끝나면(오류 포함) logs\run_exit.txt 에 종료 코드와 시각(started= 는 run_start.txt 와 같은 값)을 쓴다.
# - 이미 실행 중인 run.py 가 있으면(logs\run.lock) 아무 파일도 건드리지 않고 종료 코드 3 으로 끝낸다.
# - 예상 시간(-Stage all, 실측 기반 2026-10-19): 보통 약 1시간~1시간 30분(CatBoost 학습 라운드 130~350, 1차 실측 범위 가정),
#   최악 약 6시간 20분(모든 CatBoost 가 2,000라운드, DL 30에폭). 단계별: logs\timing_seq.json, logs\timing_seq_h30.json
# - F·G 추가분 예상 시간(실측 logs\timing_fg.json·timing_fg2.json, 시드 42 한 번: F1 77초, F2 36초, F3 38초, F4 80초, G1 50초, G2 56초, G3 39초,
#   A3+F·G 전부 94초): f 약 12~14분, g 약 7~9분, combo 약 3분(높은 묶음 없음 -> 2개 x 시드 1) ~ 약 26~35분(7묶음 모두 높음 -> 8조합 x 시드 3),
#   tables 1분 미만 -> 합계 보통 약 25분~1시간. 최악(모든 CatBoost 2,000라운드) 약 2~4시간 20분.
#   메모리 한 프로세스 private 최대 약 3.3GB(실측). 측정 중 시스템 사용 가능 메모리가 1.7GB 까지 내려간 적 있음 -> 큰 프로그램 닫고 시작.
# - M 추가분(-Stage m) 예상 시간(실측 scratch\wxanom\timing_wxanom.json, 전체 자료 시드 42 한 번: M1 87초, A3_M1 82초 - E00 34초·A3 38초보다
#   평가 때 57칸을 h 마다 새로 계산하는 시간이 더해짐): 2실험 x 시드 3 = 6실행 약 9분 + tables 1분 미만 -> 약 10분.
#   최악(모든 CatBoost 2,000라운드) 약 45분~1시간. 메모리 한 프로세스 private 최대 약 3.5GB(실측), 측정 중 시스템 사용 가능 메모리 0.6GB 까지 내려감.
# - C4·N 추가분 예상 시간(실측 scratch\c4nb\timing_c4nb.json, 전체 자료 시드 42 한 번: C4_base 38초, C4_hist 27초, A3_N1 92초, A3_N2 42초,
#   A3_N1N2 55초, A3_N0 105초): -Stage c(C4 만 남음) 2실험 x 시드 3 = 약 3~4분, -Stage n 4실험 x 시드 3 = 약 15분, tables 1분 미만.
#   최악(모든 CatBoost 2,000라운드) c 약 25분, n 약 1시간 20분.
#   메모리 한 프로세스 private 최대: c 약 3.0GB, n 약 4.1GB(A3_N1N2)·6.4GB(A3_N0, TF-IDF 500칸 - 학습 중 최대). 측정 중 시스템 사용 가능 메모리가
#   0.1~0.2GB 까지 내려감(측정 시작 때 사용 가능 5.1GB, 다른 python 프로세스가 돌던 중) -> -Stage n 은 큰 프로그램을 닫고 사용 가능 메모리 8GB 이상에서 시작 권장.
# - HX·AE 추가분 예상 시간(실측 scratch\ae\timing.json, 전체 자료 시드 42 한 번, 2026-10-19): A3_L1 38초, A3_U1 37초, A3_L1U1 39초,
#   S_AEt 193초·S_VAEt 214초·S_AEs 175초·S_VAEs 123초(= 그 모델 AE 점수 계산: 교차 적합 6모델 학습 + 검증·시험 h 1~60 점수, GPU),
#   A3_AE*·A3_*_p 8개 각 39~45초(점수는 메모리·results\_ae_scores 에서 읽음). 시드 하나 약 19분 ->
#   -Stage hx 3실험 x 시드 3 = 약 6분, -Stage ae 12실험 x 시드 3 = 약 51분, tables 1분 미만 -> 합계 약 1시간.
#   최악(모든 CatBoost 2,000라운드, AE 모든 모델 50에폭) 약 3시간 30분~4시간. AE 학습은 대부분 50에폭 상한 근처에서 끝남(표 AE·VAE).
#   메모리 한 프로세스 private 최대 약 5.0GB(CUDA 포함, 실측), GPU torch 할당 최대 0.63GB(시퀀스)·장치 전체 사용 최대 1.9GB.
#   측정 중 시스템 사용 가능 메모리가 0.43GB 까지 내려감(시작 5.7GB) -> -Stage ae 는 큰 프로그램을 닫고 사용 가능 메모리 8GB 이상에서 시작 권장.
#   AE 점수 재현성: 결정적 설정은 AE 를 같은 프로세스에서 CUDA 를 처음 쓸 때 가장 확실 -> -Stage ae 를 따로 실행 권장(다른 GPU 에서는 값이 다를 수 있음)
# - 메모리(한 프로세스 private 최대, 실측): E00·A·B·C·D 약 3GB, H10 약 7.5GB, H30 약 8.1~8.3GB, DL 약 4.5GB(GPU 약 1.5GB).
#   시작 전에 시스템 사용 가능 메모리 10GB 이상 권장(브라우저·한글 등 큰 프로그램 닫기). 멈추면 다시 실행하면 이어서 함.
param(
    [ValidateSet('all', 'e00', 'prev1', 'a_k', 'a', 'b', 'c', 'd', 'h10', 'hbig', 'h60', 'dl', 'f', 'g', 'combo', 'm', 'n', 'q', 'p', 'hx', 'ae', 'xs', 'stk', 'tables')]
    [string]$Stage = 'all',
    [switch]$Smoke,
    [string]$Only = ''
)

$here = $PSScriptRoot
$py = 'C:\Users\NT751\Desktop\천영진\리스테리아ML\.venv-exp2\Scripts\python.exe'
Set-Location $here
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
New-Item -ItemType Directory -Force -Path (Join-Path $here 'logs') | Out-Null
$consoleLog = Join-Path $here 'logs\console.log'
$exitFile = Join-Path $here 'logs\run_exit.txt'
$startFile = Join-Path $here 'logs\run_start.txt'
$lockFile = Join-Path $here 'logs\run.lock'

# 같은 실행을 두 번 띄우지 않음: 살아 있는 run.py 가 잠금을 쥐고 있으면 run_exit.txt 를 건드리지 않고 끝냄
if (Test-Path -LiteralPath $lockFile) {
    $lockPid = ''
    foreach ($ln in (Get-Content -LiteralPath $lockFile -Encoding UTF8)) { if ($ln -like 'pid=*') { $lockPid = $ln.Substring(4).Trim() } }
    if ($lockPid -match '^\d+$') {
        $p = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $lockPid) -ErrorAction SilentlyContinue
        if ($p -and $p.CommandLine -like '*run.py*') {
            Write-Host ("이미 실행 중인 run.py 가 있음 (pid {0}). 이 창은 아무것도 하지 않고 끝냄. 잠금 파일: {1}" -f $lockPid, $lockFile)
            exit 3
        }
    }
}
Remove-Item -LiteralPath $exitFile -ErrorAction SilentlyContinue

$pyArgs = @('-u', (Join-Path $here 'src\run.py'), '--stage', $Stage, '--console-log', $consoleLog)
if ($Smoke) { $pyArgs += '--smoke' }
if ($Only -ne '') { $pyArgs += @('--only', $Only) }

$start = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
[System.IO.File]::WriteAllText($startFile, ("started={0}`nargs={1}`nps_pid={2}`n" -f $start, ($pyArgs -join ' '), $PID), (New-Object System.Text.UTF8Encoding($false)))
Write-Host ("[{0}] 2차 실험 시작: {1} (python: {2})" -f $start, ($pyArgs -join ' '), $py)
$code = 1
try {
    & $py @pyArgs
    $code = $LASTEXITCODE
} catch {
    Write-Host ("실행 오류: {0}" -f $_)
    $code = 1
} finally {
    $end = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ($code -ne 3) {   # 3 = python 쪽 잠금 거절(다른 실행이 도는 중) -> 그 실행의 기록을 덮지 않음
        $line = "exit_code={0}`nstarted={1}`nfinished={2}`nargs={3}" -f $code, $start, $end, ($pyArgs -join ' ')
        [System.IO.File]::WriteAllText($exitFile, $line + "`n", (New-Object System.Text.UTF8Encoding($false)))
    }
}
if ($code -eq 0) {
    Write-Host ("[{0}] 완료. 결과표: tables\실험기록.xlsx" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
} elseif ($code -eq 3) {
    Write-Host ("[{0}] 다른 run.py 가 실행 중이라 시작하지 않음 (logs\run.lock). 그 실행의 run_exit.txt 는 건드리지 않음." -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
} else {
    Write-Host ("[{0}] 오류로 멈춤 (종료 코드 {1}). logs\progress.log, logs\console.log 확인. 고친 뒤 다시 실행하면 이어서 함." -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $code)
}
exit $code
