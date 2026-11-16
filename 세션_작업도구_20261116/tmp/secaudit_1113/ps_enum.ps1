# 읽기 전용: 디렉터리 목록(이름·크기·날짜)만 읽어 TSV 로 기록. 재분석 지점(junction·symlink) 폴더는 따라가지 않음.
param([string]$OutPath)
$roots = @('C:\Users\NT751', 'C:\Users\Public', 'C:\tmp', 'C:\Temp')
Get-ChildItem -Force -LiteralPath 'C:\' -Directory | Where-Object { $_.Name -like '%I9*' } | ForEach-Object { $roots += $_.FullName }
$enc = New-Object System.Text.UTF8Encoding($false)
$sw = New-Object System.IO.StreamWriter($OutPath, $false, $enc)
$stack = New-Object 'System.Collections.Generic.Stack[string]'
foreach ($r in $roots) { if (Test-Path -LiteralPath $r) { $stack.Push($r) } }
$rp = [System.IO.FileAttributes]::ReparsePoint
while ($stack.Count -gt 0) {
    $d = $stack.Pop()
    try { $items = (New-Object System.IO.DirectoryInfo($d)).GetFileSystemInfos() }
    catch { $sw.WriteLine("ERR`t" + $d + "`t" + $_.Exception.GetType().Name); continue }
    foreach ($it in $items) {
        if ($it -is [System.IO.DirectoryInfo]) {
            if (($it.Attributes -band $rp) -ne 0) { $sw.WriteLine("LINK`t" + $it.FullName); continue }
            $stack.Push($it.FullName)
        } else {
            $sw.WriteLine("F`t" + $it.FullName + "`t" + $it.Length + "`t" + $it.CreationTime.ToString('yyyy-MM-dd HH:mm') + "`t" + $it.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))
        }
    }
}
$sw.Close()
"done"
