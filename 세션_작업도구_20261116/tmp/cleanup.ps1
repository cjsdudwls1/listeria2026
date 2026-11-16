$ErrorActionPreference = 'Stop'
$root = 'C:\Users\NT751\Desktop\천영진'
$arc = Join-Path $root '보관\리스테리아_정리_20261105'
$moves = @(
  @('임시_외부지표_자료확인', '루트'),
  @('축사조인용_domestic.parquet', '루트'),
  @('축사조인용_overseas.parquet', '루트'),
  @('20261026_19_59_12.png', '루트'),
  @('.ipynb', '루트'),
  @('리스테리아ML\docs', '리스테리아ML'),
  @('리스테리아ML\notebook', '리스테리아ML'),
  @('리스테리아ML\scripts', '리스테리아ML'),
  @('리스테리아ML2\data\외부_데이터\CLM_SPI_DD_20261012103141.csv', '리스테리아ML2_외부_데이터'),
  @('리스테리아ML2\data\외부_데이터\CLM_SPI_DD_합본.csv', '리스테리아ML2_외부_데이터'),
  @('리스테리아ML2\data\raw\raw_t_prdlst_cl_ba_data_info.parquet', '리스테리아ML2_raw')
)
$log = @()
foreach ($m in $moves) {
  $src = Join-Path $root $m[0]
  $dstDir = Join-Path $arc $m[1]
  if (-not (Test-Path -LiteralPath $src)) { $log += "없음: $($m[0])"; continue }
  New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
  Move-Item -LiteralPath $src -Destination $dstDir
  $dst = Join-Path $dstDir (Split-Path $m[0] -Leaf)
  if ((Test-Path -LiteralPath $dst) -and -not (Test-Path -LiteralPath $src)) { $log += "옮김: $($m[0]) -> 보관\리스테리아_정리_20261105\$($m[1])" } else { $log += "실패: $($m[0])" }
}
# 쓰레기 캐시는 휴지통으로
Add-Type -AssemblyName Microsoft.VisualBasic
$pc = Join-Path $root '리스테리아ML2\notebook\__pycache__'
if (Test-Path -LiteralPath $pc) {
  [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($pc, 'OnlyErrorDialogs', 'SendToRecycleBin')
  $log += "휴지통: 리스테리아ML2\notebook\__pycache__"
}
$log | Out-File -FilePath 'C:\Users\NT751\.claude\jobs\a19471fc\tmp\cleanup_log.txt' -Encoding utf8
