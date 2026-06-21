import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TRAFFIC_API")
print(f"Traffic API Key: {API_KEY}")

def get_congestion(lat, lon):
    url = (
        "https://api.tomtom.com/traffic/services/4/"
        f"flowSegmentData/absolute/10/json?point={lat},{lon}&key={API_KEY}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()["flowSegmentData"]

    current_speed = data["currentSpeed"]
    free_flow_speed = data["freeFlowSpeed"]

    if free_flow_speed == 0:
        return 0

    congestion = 1 - (current_speed / free_flow_speed)

    return round(max(0, min(1, congestion)), 4)


if __name__ == "__main__":
    # Only runs when executing this file directly, not on import
    lat = 12.9716
    lon = 77.5946
    score = get_congestion(lat, lon)
    print(f"Congestion Score: {score}")
    print(f"Congestion Percent: {score * 100:.2f}%")
