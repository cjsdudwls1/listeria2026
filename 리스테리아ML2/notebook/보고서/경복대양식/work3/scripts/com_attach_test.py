# -*- coding: utf-8 -*-
"""한글 COM 시험: 사람이 띄운 한글(-Automation 없이 실행)이 떠 있을 때 조립기처럼 새 COM 객체(HWPFrame.HwpObject)를 만들면
새 한글 프로세스(hwp.exe -Automation)가 따로 뜨는지, 이미 떠 있는 한글에 붙는지 확인 — 조립기가 사용자 한글 창을 닫을 수 있는지.

안전장치: 시작할 때 한글이 하나라도 떠 있으면 바로 멈춤(사용자 한글을 건드리지 않음). 이 시험이 띄운 PID 만 끝냄.
시험 중에는 보고서/hwp_build.lock 을 잡음(조립기와 같은 함수).
"""
import csv
import io
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:/Users/NT751/Desktop/천영진/리스테리아ML2/notebook/보고서/경복대양식")
import build_kb_hwp as B  # noqa: E402

HWP_EXE = r"C:\Program Files (x86)\Hnc\Office 2020\HOffice110\Bin\hwp.exe"


def procs():
    """{PID: (창 제목, 명령줄)}"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='Hwp.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
                       capture_output=True, encoding="utf-8", errors="replace")
    titles = B.hwp_processes()
    out = {}
    for line in r.stdout.splitlines():
        if "\t" in line:
            pid, cmd = line.split("\t", 1)
            pid = int(pid)
            out[pid] = (titles.get(pid, "?"), cmd.strip())
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p0 = procs()
    if p0:
        raise SystemExit("한글이 이미 떠 있어 시험하지 않음(사용자 것일 수 있음): %s" % p0)
    B.lock_acquire()
    mine = set()
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 7                                  # SW_SHOWMINNOACTIVE: 최소화, 포커스 안 뺏음
        p = subprocess.Popen([HWP_EXE], startupinfo=si)
        mine.add(p.pid)
        print("사람처럼 띄운 한글(인자 없음) PID", p.pid)
        t0 = time.time()
        while time.time() - t0 < 25 and p.pid not in procs():
            time.sleep(1)
        time.sleep(6)
        before = procs()
        print("COM 만들기 전:", before)
        unknown = [k for k in before if k not in mine]
        if unknown:
            raise SystemExit("시험 중 다른 한글이 떠서 멈춤: %s" % unknown)
        import pythoncom
        import win32com.client as win32
        pythoncom.CoInitialize()
        obj = win32.DispatchEx("HWPFrame.HwpObject")
        time.sleep(4)
        after = procs()
        new = [k for k in after if k not in before]
        mine.update(k for k in new if "-Automation" in after[k][1])
        print("COM 만든 뒤:", after)
        if new:
            print("결과: 새 한글 프로세스가 따로 뜸 %s -> 조립기는 사람이 띄운 한글에 붙지 않음" % new)
        else:
            print("결과: 새 프로세스 없음 -> COM 객체가 이미 떠 있던 한글(PID %s)에 붙음 -> 조립기 Quit 이 그 한글을 닫을 수 있음"
                  % list(before))
        try:
            obj.Clear(1)
            obj.Quit()                                       # 이 시험이 만든 객체(새 프로세스든, 시험이 띄운 한글이든)
        except Exception as e:
            print("Quit 오류", e)
        del obj
        time.sleep(3)
        print("Quit 뒤:", procs())
    finally:
        now = procs()
        for pid in mine:
            if pid in now:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                print("시험이 띄운 한글 종료 PID", pid)
        B.lock_release()
        print("끝:", procs())


if __name__ == "__main__":
    main()
