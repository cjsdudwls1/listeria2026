# -*- coding: utf-8 -*-
"""DB 접속을 2분 간격으로 재시도하고, 되면 share_collect_db.py 를 실행"""
import sys, time, subprocess, datetime as dt
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import psycopg
from db_utils import pgpass_entry
e = pgpass_entry()
for i in range(40):
    try:
        c = psycopg.connect(host=e['host'], port=e['port'], dbname=e['dbname'], user=e['user'], options='-c default_transaction_read_only=on', connect_timeout=20, application_name='listeria_collect')
        c.close(); print(dt.datetime.now().strftime('%H:%M:%S'), '접속 성공 → 수집 시작', flush=True)
        r = subprocess.run([sys.executable, r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\share_collect_db.py"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\share_collect_db.log", 'w', encoding='utf-8').write(r.stdout + '\n' + r.stderr)
        print('수집 종료 코드', r.returncode, flush=True); sys.exit(r.returncode)
    except Exception as ex:
        print(dt.datetime.now().strftime('%H:%M:%S'), f'시도 {i+1}/40 실패:', type(ex).__name__, flush=True); time.sleep(120)
print('DB 접속 실패로 종료'); sys.exit(2)
