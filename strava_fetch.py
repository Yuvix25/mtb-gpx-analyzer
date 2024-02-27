import os
import sys
import json
import requests
from typing import List, Tuple, Dict

SEGMENTS_FILE = "segments.json"

DATA_URL = "https://www.strava.com/stream/segments/{}"

class Segment:
    def __init__(self, id: str, name: str, latlng: List[Tuple[float, float]], distance: List[float], altitude: List[float]):
        self.id = id
        self.name = name
        self.latlng = latlng
        self.distance = distance
        self.altitude = altitude


def load_segments() -> Dict[str, Segment]:
    if not os.path.exists(SEGMENTS_FILE):
        return {}
    with open(SEGMENTS_FILE, "r") as f:
        data = json.load(f)
        return {id: Segment(id, **obj) for id, obj in data.items()}



def fetch_segment(segment_id, name):
    response = requests.get(DATA_URL.format(segment_id))
    data = response.json()
    data["name"] = name

    segments = load_segments()
    segments[segment_id] = data
    with open(SEGMENTS_FILE, "w") as f:
        json.dump(segments, f)

if __name__ == "__main__":
    fetch_segment(sys.argv[1], " ".join(sys.argv[2:]))