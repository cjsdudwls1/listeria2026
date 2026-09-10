import sys
sys.stdout.reconfigure(encoding='utf-8')
from db_utils import connect
conn = connect(readonly=False)
with conn.cursor() as cur:
    cur.execute("SELECT has_database_privilege(current_database(),'TEMP') AS temp_ok, has_database_privilege(current_database(),'CREATE') AS db_create")
    print("db privs:", cur.fetchone())
    cur.execute("""SELECT nspname, has_schema_privilege(nspname,'CREATE') AS can_create, has_schema_privilege(nspname,'USAGE') AS can_use
                   FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' AND nspname<>'information_schema' ORDER BY 1""")
    for r in cur.fetchall(): print("  ", r)
    cur.execute("SELECT rolsuper, rolcreatedb FROM pg_roles WHERE rolname=current_user"); print("role:", cur.fetchone())
    cur.execute("SHOW statement_timeout"); print("statement_timeout:", cur.fetchone())
    cur.execute("SHOW work_mem"); print("work_mem:", cur.fetchone())
    cur.execute("SHOW max_parallel_workers_per_gather"); print("parallel:", cur.fetchone())
    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('ready_data.hazard_preprocessed_data'))"); print("body size:", cur.fetchone())
conn.close()
