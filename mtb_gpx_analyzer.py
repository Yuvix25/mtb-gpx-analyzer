from __future__ import annotations
from typing import List, Tuple, Literal, Optional

import argparse
import gpxpy
import gpxpy.gpx
from geopy import distance
import datetime
from datetime import timedelta
from dateutil import tz
import numpy as np
import matplotlib.pyplot as plt

from strava_fetch import load_segments


def closest_on_line(line, p3):
    p1, p2 = line
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return x1+a*dx, y1+a*dy

def dot(v, w):
    """
    row-wise dot product of 2-dimensional arrays
    """
    return np.einsum('ij,ij->i', v, w)


def closest_to_lines(line_starts, line_ends, p):
    """
    find line segment closest to the point p 
    """

    # array of vectors from the start to the end of each line segment
    se = line_ends - line_starts
    # array of vectors from the start of each line segment to the point p
    sp = p - line_starts
    # array of vectors from the end of each line segment to p
    ep = p - line_ends

    # orthogonal projection of sp onto se
    proj = (dot(sp, se) / dot(se, se)).reshape(-1, 1) * se
    # orthogonal complement of the projection
    n = sp - proj
    
    # squares of distances from the start of each line segment to p
    starts_d = dot(sp, sp)
    # squares of distances from the end of each line segments to p
    ends_d = dot(ep, ep)
    # squares of distances between p and each line
    lines_d = dot(n, n)

    # If the point determined by the projection is inside
    # the line segment, it is the point of the line segment
    # closest to p; otherwhise the closest point is one of
    # the enpoints. Determine which of these cases holds 
    # and compute the square of the distance to each line segment. 
    coeffs = dot(proj, se)
    dist = np.select([coeffs < 0, coeffs < dot(se, se), True],
                     [starts_d, lines_d, ends_d])

    # find the index of the closest line segment, its distance to p,
    # and the point in this line segment closest to p
    idx = np.argmin(dist)
    min_dist = dist[idx]
    if min_dist == starts_d[idx]:
        min_point = line_starts[idx]
    elif min_dist == ends_d[idx]:
        min_point = line_ends[idx]
    else:
        min_point = line_starts[idx] + proj[idx]
    return idx, min_dist**0.5, min_point

def closest_to_lines_simple(lines, p):
    line_starts = np.array(lines[:-1])
    line_ends = np.array(lines[1:])
    return closest_to_lines(line_starts, line_ends, p)

class GeoPoint:
    CLOSE_THRESHOLD = 30 # meters
    def __init__(self, type: Literal['start', 'end'], latitude, longitude):
        self.geo_point = (latitude, longitude)
        self.type = type

        self.potential_results: Optional[List[Tuple[float, gpxpy.gpx.GPXTrackPoint]]] = None
        self.line_starts = []
        self.line_ends = []

    def search(self, point: gpxpy.gpx.GPXTrackPoint):
        dist = distance.distance((point.latitude, point.longitude), self.geo_point).meters

        if dist <= GeoPoint.CLOSE_THRESHOLD:
            entry = (dist, point)
            if self.potential_results is None:
                self.potential_results = [entry]
            else:
                self.potential_results.append(entry)
            self.line_starts.append([point.latitude, point.longitude])
            self.line_ends.append([point.latitude, point.longitude])
            return True
        elif self.potential_results is not None:
            return self.get_best()
        return False
    
    def get_best(self):
        try:
            if len(self.potential_results) == 1:
                return self.potential_results[0][1]

            self.line_starts.pop(-1)
            self.line_ends.pop(0)
            
            line_starts = np.array(self.line_starts)
            line_ends = np.array(self.line_ends)

            i, _, _ = closest_to_lines(line_starts, line_ends, np.array(self.geo_point))

            point = closest_on_line((line_starts[i], line_ends[i]), self.geo_point)
            dist_before = distance.distance(line_starts[i], point).m
            dist_after = distance.distance(line_ends[i], point).m
            before = TrailPoint.from_gpx(self.potential_results[i][1])
            after = TrailPoint.from_gpx(self.potential_results[i+1][1])
            after.distance = after.distance_to(before)
            extra_point = [TrailRun.point_lerp(before, after, dist_before, dist_after)]
            if self.type == 'start':
                res = extra_point + list(map(lambda x: x[1], self.potential_results[i+1:]))
            else:
                res = list(map(lambda x: x[1], self.potential_results[:i+1])) + extra_point

            return res
        finally:
            self.line_starts = []
            self.line_ends = []
            self.potential_results = None

        


def replace_tzinfo(dt: datetime.datetime):
    utc_offset = dt.utcoffset()
    if utc_offset is not None:
        dt = dt.replace(tzinfo=datetime.timezone(utc_offset))
    return dt.astimezone(tz.tzlocal())


class TrailPointDelta:
    def __init__(self, distance: float, latitude: float, longitude: float, elevation: float, time: timedelta, speed: float):
        self.distance = distance
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.time = time
        self.speed = speed
    
    def __sub__(self, other: TrailPointDelta):
        return TrailPointDelta(
            self.distance - other.distance,
            self.latitude - other.latitude,
            self.longitude - other.longitude,
            self.elevation - other.elevation,
            self.time - other.time,
            self.speed - other.speed,
        )
    
    def __neg__(self):
        return TrailPointDelta(
            -self.distance,
            -self.latitude,
            -self.longitude,
            -self.elevation,
            -self.time,
            -self.speed,
        )
    
    def __add__(self, other: TrailPoint | TrailPointDelta):
        if isinstance(other, TrailPoint):
            return TrailPoint(
                other.extensions,
                self.distance,
                other.latitude + self.latitude,
                other.longitude + self.longitude,
                other.elevation + self.elevation,
                other.time + self.time,
                None,
                None,
                None,
                None,
                None,
                other.speed + self.speed,
                None,
            )
        return self - (-other)
    
    def __mul__(self, other: float):
        return TrailPointDelta(
            self.distance * other,
            self.latitude * other,
            self.longitude * other,
            self.elevation * other,
            self.time * other,
            self.speed * other,
        )
    
    def __truediv__(self, other: float):
        return TrailPointDelta(
            self.distance / other,
            self.latitude / other,
            self.longitude / other,
            self.elevation / other,
            self.time / other,
            self.speed / other,
        )

class TrailPoint(gpxpy.gpx.GPXTrackPoint):
    def __init__(self, extensions, distance, latitude, longitude, elevation, time, symbol, comment, horizontal_dilution, vertical_dilution, position_dilution, speed, name):
        super().__init__(latitude, longitude, elevation, time, symbol, comment, horizontal_dilution, vertical_dilution, position_dilution, speed, name)
        self.time = replace_tzinfo(self.time)
        self.distance = distance
        if self.speed == None:
            self.speed = 0
        self.extensions = []
        for ext in extensions:
            self.extensions.append(ext)

    @staticmethod
    def from_gpx(parent: gpxpy.gpx.GPXTrackPoint):
        point = TrailPoint(parent.extensions, 0, parent.latitude, parent.longitude, parent.elevation, parent.time, parent.symbol, parent.comment, parent.horizontal_dilution, parent.vertical_dilution, parent.position_dilution, parent.speed, parent.name)
        return point

    def distance_to(self, other: TrailPoint|GeoPoint):
        self_geo = (self.latitude, self.longitude)
        if isinstance(other, TrailPoint):
            return distance.distance(self_geo, (other.latitude, other.longitude)).m
        return distance.distance(self_geo, other.geo_point).m
    
    def time_to(self, other: TrailPoint) -> timedelta:
        return self.time - other.time
    
    def speed_to(self, other: TrailPoint) -> float:
        if self.speed is None or other.speed is None:
            return 0
        return self.speed - other.speed
    
    def elevation_to(self, other: TrailPoint) -> float:
        return self.elevation - other.elevation
    
    def heartrate_to(self, other: TrailPoint) -> float:
        return self.heartrate - other.heartrate
    
    @property
    def heartrate(self) -> float:
        if len(self.extensions) == 0:
            return 0
        if len(self.extensions[0]) == 0:
            return 0
        if not self.extensions[0][0].tag.endswith('hr'):
            return 0
        return float(self.extensions[0][0].text)

    def __sub__(self, other: TrailPoint):
        return TrailPointDelta(
            self.distance - other.distance,
            self.latitude - other.latitude,
            self.longitude - other.longitude,
            self.elevation - other.elevation,
            self.time - other.time,
            self.speed_to(other),
        )
    
    def __add__(self, other: TrailPointDelta):
        return TrailPoint(
            self.extensions,
            self.distance + other.distance,
            self.latitude + other.latitude,
            self.longitude + other.longitude,
            self.elevation + other.elevation,
            self.time + other.time,
            None,
            None,
            None,
            None,
            None,
            self.speed + other.speed,
            None,
        )
    
    def __str__(self) -> str:
        return f'({self.latitude}, {self.longitude})\t| {self.time}'
    
    def __repr__(self) -> str:
        return str(self)
    


class TrailRun:
    PLT_COLOR_COUNTER = 0

    def __init__(self, trail: Trail):
        self.trail = trail
        self.points: List[TrailPoint] = []

    def add_point(self, point: gpxpy.gpx.GPXTrackPoint):
        if point.time is None:
            return False
        point = TrailPoint.from_gpx(point)
        point = self.trail.closest_on_trail(point)
        
        if len(self.points) >= 1:
            prev_point = self.points[-1]
            delta_distance = point.distance - prev_point.distance
            delta_time = point.time_to(prev_point)
            if point.speed == 0:
                point.speed = 3.6 * delta_distance / delta_time.total_seconds()
            
            if point.speed < 0:
                return False
        
        if len(self.points) == 1:
            self.points[0].speed = point.speed

        self.points.append(point)
    
    def add_points(self, points: List[gpxpy.gpx.GPXTrackPoint]):
        for point in points:
            self.add_point(point)
    
    def point_at(self, distance):
        n = len(self.points)
        low, high = 0, n-1
 
        while low <= high:
            mid = (high + low) // 2
    
            if self.points[mid].distance < distance:
                low = mid + 1
            elif self.points[mid].distance > distance:
                high = mid - 1
            else:
                return self.points[mid]
        
        mid = max(0, min(n-1, mid))
        if self.points[mid].distance < distance and mid < n - 1:
            a = self.points[mid]
            b = self.points[mid + 1]
        elif self.points[mid].distance > distance and mid > 0:
            a = self.points[mid - 1]
            b = self.points[mid]
        else:
            return self.points[mid]

        return TrailRun.point_lerp(a, b, distance - a.distance, b.distance - distance)
    
    def finalize(self):
        # make sure we never go backwards in the start/end of the run (trim)
        for i in range(1, len(self.points)):
            if self.points[i-1].distance < self.points[i].distance:
                self.points = self.points[i-1:]
                break

        for i in range(len(self.points)-2, -1, -1):
            if self.points[i].distance < self.points[i+1].distance:
                self.points = self.points[:i+2]
                break
    
    @staticmethod
    def point_lerp(point1: TrailPoint, point2: TrailPoint, distance_before: float, distance_after: float) -> TrailPoint:
        points_distance = point2.distance - point1.distance
        if distance_after > points_distance:
            distance_before = -distance_before

        frac = distance_before / points_distance
        return point1 + (point2-point1) * frac


    @property
    def total_time(self) -> timedelta:
        if len(self.points) == 0:
            return timedelta()
        return self.points[-1].time_to(self.points[0])
    
    def plot(self, index, total, ax, time_ax, heart_rate_ax, label: str):
        dist = []
        speed = []
        time = []
        hr = []
        for point in self.points:
            dist.append(point.distance)
            speed.append(point.speed)
            current_time = point.time_to(self.points[0])
            if total == 2:
                if index == 1:
                    point_other = self.trail.runs[0].point_at(point.distance)
                    time.append((current_time - point_other.time_to(self.trail.runs[0].points[0])).total_seconds())
            else:
                time.append(current_time.total_seconds())
            hr.append(point.heartrate)
        a = ax.plot(dist, speed, 'C' + str(TrailRun.PLT_COLOR_COUNTER), label=label)
        if len(time) > 0:
            b = time_ax.plot(dist, time, 'C' + str(TrailRun.PLT_COLOR_COUNTER+1), label=label + (('-' + str(self.trail.runs[0].total_time)) if total == 2 else ''), linestyle='dashed')
        else:
            b = []
        c = heart_rate_ax.plot(dist, hr, 'C' + str(TrailRun.PLT_COLOR_COUNTER+2), label=label, linestyle='dotted')
        TrailRun.PLT_COLOR_COUNTER += 3

        return a + b + c
    
    def to_string(self, run_id=None, minimized=False) -> str:
        if run_id is None:
            run_id = ''
        else:
            run_id = ' ' + str(run_id)
        return  f'Run{run_id} - {self.trail.name}:' + \
                ((f'\nStart\t{self.points[0]}\n' + \
                f'End\t{self.points[-1]}\nTotal\t') if not minimized else f' {self.points[0].time.strftime("%Y-%m-%d %H:%M:%S")} | ') + \
                str(self.total_time)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return str(self)


class Trail:
    trails: List[Trail] = []

    @staticmethod
    def all_runs() -> List[TrailRun]:
        runs = []
        for trail in Trail.trails:
            for run in trail.runs:
                runs.append(run)
        return runs

    def __init__(self, name: str, points: List[Tuple[float, float]], distance: List[float], elevation: List[float]):
        self.name = name
        self.points = points
        self.start = GeoPoint('start', *points[0])
        self.end = GeoPoint('end', *points[-1])
        self.distance = distance
        self.elevation = elevation

        self.runs: List[TrailRun] = []

        self.loading_run = None
        self.last_distance = None
        
        Trail.trails.append(self)

    def closest_on_trail(self, point: TrailPoint):
        point_geo = (point.latitude, point.longitude)
        # find the closest point to the line
        i, _, _ = closest_to_lines_simple(self.points, point_geo)

        closest = closest_on_line((self.points[i], self.points[i+1]), point_geo)
        dist_before = distance.distance(self.points[i], closest).m
        dist_after = distance.distance(self.points[i+1], closest).m
        before = TrailPoint(point.extensions, self.distance[i], self.points[i][0], self.points[i][1], self.elevation[i], point.time, None, None, None, None, None, 0, None)
        after = TrailPoint(point.extensions, self.distance[i+1], self.points[i+1][0], self.points[i+1][1], self.elevation[i], point.time, None, None, None, None, None, 0, None)
        res = TrailRun.point_lerp(before, after, dist_before, dist_after)
        return res


    def _visit(self, point: gpxpy.gpx.GPXTrackPoint):
        start = self.start.search(point)
        end = self.end.search(point)

        if isinstance(start, list):
            self.loading_run = TrailRun(self)
            self.loading_run.add_points(start)
        elif self.loading_run is not None:
            if isinstance(end, list):
                self.loading_run.add_points(end)
                self.loading_run.finalize()
                self.runs.append(self.loading_run)
                self.loading_run = None
            elif end == False:
                self.loading_run.add_point(point)

    @staticmethod
    def visit(point: gpxpy.gpx.GPXTrackPoint):
        TrailPoint.from_gpx(point) # time is sometimes broken

        for trail in Trail.trails:
            trail._visit(point)

    @staticmethod
    def filter_runs(choice: List[int]) -> List[TrailRun]:
        all_runs = Trail.all_runs()
        return [all_runs[i] for i in choice]

    @staticmethod
    def plot(runs: List[TrailRun] = None):
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.75)

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Speed (km/h)')

        time_twin = ax.twinx()
        hr_twin = ax.twinx()
        hr_twin.spines.right.set_position(('axes', 1.2))

        hr_twin.set_ylabel('Heart Rate (bpm)')

        time_twin.set_ylabel('Time (s)')

        handles = []
        for trail in Trail.trails:
            filtered = list(filter(lambda x: runs is None or x in runs, trail.runs))
            for i, run in enumerate(filtered):
                new_handles = run.plot(i, len(filtered), ax, time_twin, hr_twin, str(trail.name) + ' - ' + str(run.total_time))
                handles = handles + new_handles

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        # Put a legend to the right of the current axis
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.4, 0.5))

        plt.show()


if __name__ == '__main__':
    segments = load_segments()

    for segment_id, segment in segments.items():
        Trail(segment.name, segment.latlng, segment.distance, segment.altitude)
    
    parser = argparse.ArgumentParser(description='Analyze GPX files')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='GPX files to analyze')
    args = parser.parse_args()

    files = args.files

    for file in files:
        gpx_file = open(file, 'r', encoding='utf-8')
        gpx = gpxpy.parse(gpx_file)

        points = []
        line_starts = []
        line_ends = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    Trail.visit(point)

    all_runs = Trail.all_runs()
    while True:
        for i, run in enumerate(all_runs):
            print(run.to_string(i+1, True))
        inp = input('Analyze runs (comma separated): ').strip()
        print()
        if len(inp) == 0:
            choice = list(range(len(all_runs)))
        else:
            choice = list(map(lambda x: int(x.strip()) - 1, inp.split(',')))

        runs = Trail.filter_runs(choice)
        for i, run in zip(choice, runs):
            print(run.to_string(i+1) + '\n')

        Trail.plot(runs)
