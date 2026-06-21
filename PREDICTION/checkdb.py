from database import get_conn

conn = get_conn()
try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        print(cur.fetchall())
finally:
    conn.close()
