from flask import Flask, jsonify
from flask_cors import CORS
import os
from database import get_conn, init_db

app = Flask(__name__)

frontend_origin = os.environ.get("FRONTEND_ORIGIN")
if frontend_origin:
    CORS(app, origins=[frontend_origin])
else:
    CORS(app)


@app.route("/traffic")
def traffic():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM traffic_predictions")
            rows = cur.fetchall()
            # cur.description gives column names
            cols = [desc[0] for desc in cur.description]
            result = [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()

    return jsonify(result)


if __name__ == "__main__":
    init_db()
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
