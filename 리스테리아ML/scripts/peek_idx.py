import sys
sys.stdout.reconfigure(encoding='utf-8')
from db_utils import connect
conn = connect()
with conn.cursor() as cur:
    cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='ready_data' AND tablename='hazard_preprocessed_data'")
    print("--- indexes on hazard_preprocessed_data ---")
    for r in cur.fetchall(): print("  ", r[0], "|", r[1][:150])
    for t in [('ba_data','m_org_lims_data_info'),('ba_data','s_org_lims_data_info'),('ba_data','t_prdlst_mnf_ledg'),('ba_data','t_bsn_lcns_ledg'),('api_data_info','kma_asos_daily_info'),('api_data_info','nas_agri_weather_daily_info'),('api_data_info','soil_ph_info'),('api_data_info','keco_airkorea_fine_dust_info')]:
        cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname=%s AND tablename=%s", t)
        rows = cur.fetchall()
        print(f"--- {t[0]}.{t[1]}: {len(rows)} indexes ---")
        for r in rows: print("  ", r[0], "|", r[1][:150])
    cur.execute("SELECT pid, state, now()-query_start AS elapsed, left(query,120) FROM pg_stat_activity WHERE application_name='listeria_collect' AND state <> 'idle'")
    print("--- running ---")
    for r in cur.fetchall(): print("  ", r)
conn.close()
