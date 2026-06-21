from database import get_conn


def save_prediction(
    grid_id,
    timestamp,
    lat_grid,
    lon_grid,
    traffic_volume,
    number_vehicle,
    type_score,
    violation_score,
    traffic_live_score,
    final_score,
):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO traffic_predictions (
                        grid_id, timestamp, lat_grid, lon_grid,
                        traffic_volume, number_vehicle, type_score,
                        violation_score, traffic_live_score, final_score
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (grid_id) DO UPDATE SET
                        timestamp          = EXCLUDED.timestamp,
                        lat_grid           = EXCLUDED.lat_grid,
                        lon_grid           = EXCLUDED.lon_grid,
                        traffic_volume     = EXCLUDED.traffic_volume,
                        number_vehicle     = EXCLUDED.number_vehicle,
                        type_score         = EXCLUDED.type_score,
                        violation_score    = EXCLUDED.violation_score,
                        traffic_live_score = EXCLUDED.traffic_live_score,
                        final_score        = EXCLUDED.final_score
                """, (
                    str(grid_id),
                    str(timestamp),
                    float(lat_grid),
                    float(lon_grid),
                    float(traffic_volume),
                    float(number_vehicle),
                    float(type_score),
                    float(violation_score),
                    float(traffic_live_score),
                    float(final_score),
                ))
    finally:
        conn.close()
