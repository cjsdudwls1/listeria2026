"""information_schema.columns 만 조회 (읽기 전용 세션). 접속 정보는 출력하지 않음."""
import os, sys
from dotenv import dotenv_values
import psycopg2

sys.stdout.reconfigure(encoding='utf-8')
env = dotenv_values(r'C:\Users\NT751\Desktop\천영진\리스테리아ML\.env')
kw = dict(host=env.get('DB_HOST'), port=env.get('DB_PORT') or 5432, dbname=env.get('DB_NAME'), user=env.get('DB_USER'),
          connect_timeout=15, application_name='listeria_final_s0_schema',
          options='-c default_transaction_read_only=on -c statement_timeout=60000')
if env.get('DB_PASSWORD'):
    kw['password'] = env['DB_PASSWORD']
print('채워진 키:', {k: bool(env.get(k)) for k in ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']})
TABLES = [('ba_data', 's_org_lims_data_info'), ('ready_data', 'hazard_preprocessed_data'), ('ba_data_info', 'nation_code'),
          ('api_data_info', 'kma_sigungu_point'), ('api_data_info', 'global_soil_ph'), ('api_data_info', 'global_weather_info'),
          ('api_data_info', 'global_factory_weather'), ('predict_data', 'p0830_fcst_long'), ('api_data_info', 'kma_weather_predict_info'),
          ('staging_data', 'hazard_preprocessed_data')]
conn = psycopg2.connect(**kw)
conn.set_session(readonly=True, autocommit=False)
cur = conn.cursor()
cur.execute('show transaction_read_only'); print('read_only:', cur.fetchone()[0])
for s, t in TABLES:
    cur.execute("""select c.ordinal_position, c.column_name, c.data_type
                   from information_schema.columns c where c.table_schema=%s and c.table_name=%s order by 1""", (s, t))
    rows = cur.fetchall()
    print(f'\n## {s}.{t} ({len(rows)} cols)')
    for r in rows:
        print(f'{r[0]:>3} {r[1]:32s} {r[2]:28s} ')
conn.rollback(); conn.close()
