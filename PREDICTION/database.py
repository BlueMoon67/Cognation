import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    """Return a new Supabase/PostgreSQL connection (plain cursor, not RealDict)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL env var is not set")
    return psycopg2.connect(url)


def init_db():
    """Create the traffic_predictions table if it does not exist."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS traffic_predictions (
                        grid_id            TEXT PRIMARY KEY,
                        timestamp          TEXT,
                        lat_grid           DOUBLE PRECISION,
                        lon_grid           DOUBLE PRECISION,
                        traffic_volume     DOUBLE PRECISION,
                        number_vehicle     DOUBLE PRECISION,
                        type_score         DOUBLE PRECISION,
                        violation_score    DOUBLE PRECISION,
                        traffic_live_score DOUBLE PRECISION,
                        final_score        DOUBLE PRECISION
                    )
                """)
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database ready")
