import sys
sys.stdout.reconfigure(encoding='utf-8')
from db_utils import connect, pgpass_entry
e = pgpass_entry()
print("접속 대상:", {k: e[k] for k in ("host", "port", "dbname")}, "user=" + e["user"][:2] + "***")
conn = connect()
with conn.cursor() as cur:
    cur.execute("SELECT version(), current_user, current_database(), current_setting('default_transaction_read_only')")
    v, u, d, ro = cur.fetchone()
    print("server:", v.split(",")[0], "| db:", d, "| read_only:", ro)
    cur.execute("""
        SELECT table_schema, table_type, COUNT(*) 
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
        GROUP BY 1,2 ORDER BY 1,2""")
    print("--- schemas ---")
    for r in cur.fetchall(): print("  ", r)
    cur.execute("""
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema IN ('train_data','public','ready_data')
        ORDER BY 1,2""")
    print("--- train_data / public / ready_data 객체 ---")
    for r in cur.fetchall(): print("  ", r)
conn.close()
