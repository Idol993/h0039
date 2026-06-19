from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import gpxpy
import gpxpy.gpx
from geopy.distance import geodesic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()

STOP_SPEED_KMH = 1.0
DRIFT_SPEED_KMH = 200.0
STOP_DURATION_THRESHOLD = timedelta(minutes=5)
STAY_DURATION_THRESHOLD = timedelta(minutes=15)
SEGMENT_GAP_THRESHOLD = timedelta(minutes=30)
STOP_TO_MOVE_THRESHOLD = timedelta(minutes=10)


@dataclass
class TrackPoint:
    time: datetime
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    speed_kmh: float = 0.0
    distance_m: float = 0.0
    is_drift: bool = False
    is_stop: bool = False

    @property
    def coord(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)


@dataclass
class StayPoint:
    location: Tuple[float, float]
    start_time: datetime
    end_time: datetime
    duration: timedelta = field(init=False)
    name: str = "停留/办事"

    def __post_init__(self):
        self.duration = self.end_time - self.start_time

    @property
    def center_lat(self) -> float:
        return self.location[0]

    @property
    def center_lon(self) -> float:
        return self.location[1]


@dataclass
class TripSegment:
    points: List[TrackPoint] = field(default_factory=list)
    stay_points: List[StayPoint] = field(default_factory=list)
    segment_id: int = 0

    @property
    def start_time(self) -> Optional[datetime]:
        return self.points[0].time if self.points else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.points[-1].time if self.points else None

    @property
    def start_point(self) -> Optional[Tuple[float, float]]:
        if self.points:
            return (self.points[0].latitude, self.points[0].longitude)
        return None

    @property
    def end_point(self) -> Optional[Tuple[float, float]]:
        if self.points:
            return (self.points[-1].latitude, self.points[-1].longitude)
        return None

    @property
    def total_distance_km(self) -> float:
        return sum(p.distance_m for p in self.points) / 1000.0

    @property
    def duration(self) -> timedelta:
        if not self.points:
            return timedelta(0)
        return self.points[-1].time - self.points[0].time

    @property
    def avg_speed_kmh(self) -> float:
        dur_h = self.duration.total_seconds() / 3600.0
        if dur_h <= 0:
            return 0.0
        return self.total_distance_km / dur_h

    @property
    def moving_distance_km(self) -> float:
        return sum(p.distance_m for p in self.points if not p.is_stop) / 1000.0


@dataclass
class TripResult:
    file_path: str
    segments: List[TripSegment] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_duration: timedelta = timedelta(0)
    moving_time: timedelta = timedelta(0)
    employee_name: str = ""

    def recompute_totals(self):
        self.total_distance_km = sum(s.total_distance_km for s in self.segments)
        if self.segments:
            all_starts = [s.start_time for s in self.segments if s.start_time]
            all_ends = [s.end_time for s in self.segments if s.end_time]
            if all_starts and all_ends:
                self.total_duration = max(all_ends) - min(all_starts)


class TripParser:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse_gpx(self, file_path: str) -> List[TrackPoint]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"GPX file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        raw_points: List[TrackPoint] = []

        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.time is None:
                        continue
                    raw_points.append(TrackPoint(
                        time=point.time,
                        latitude=point.latitude,
                        longitude=point.longitude,
                        elevation=point.elevation,
                    ))

        for waypoint in gpx.waypoints:
            pass

        if self.verbose:
            console.print(f"[dim]读取原始轨迹点: {len(raw_points)} 个[/dim]")

        return raw_points

    def _compute_speeds_and_distances(self, points: List[TrackPoint]) -> List[TrackPoint]:
        if len(points) < 2:
            return points

        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            dist_m = geodesic(prev.coord, curr.coord).meters
            time_delta = (curr.time - prev.time).total_seconds()
            curr.distance_m = dist_m
            if time_delta > 0:
                curr.speed_kmh = (dist_m / 1000.0) / (time_delta / 3600.0)
            else:
                curr.speed_kmh = 0.0
        return points

    def _filter_drift_points(self, points: List[TrackPoint]) -> List[TrackPoint]:
        filtered = []
        removed = 0
        for p in points:
            if p.speed_kmh > DRIFT_SPEED_KMH:
                p.is_drift = True
                removed += 1
                if filtered:
                    continue
            filtered.append(p)
        if self.verbose and removed > 0:
            console.print(f"[yellow]过滤GPS漂移点: {removed} 个 (速度 > {DRIFT_SPEED_KMH} km/h)[/yellow]")
        return filtered

    def _mark_stop_points(self, points: List[TrackPoint]) -> List[TrackPoint]:
        if not points:
            return points
        n = len(points)
        stop_flags = [False] * n

        i = 0
        while i < n:
            if points[i].speed_kmh < STOP_SPEED_KMH:
                j = i
                while j < n and points[j].speed_kmh < STOP_SPEED_KMH:
                    j += 1
                if j > i:
                    duration = points[j - 1].time - points[i].time
                    if duration >= STOP_DURATION_THRESHOLD:
                        for k in range(i, j):
                            stop_flags[k] = True
                i = j
            else:
                i += 1

        for i, is_stop in enumerate(stop_flags):
            points[i].is_stop = is_stop

        stop_count = sum(1 for f in stop_flags if f)
        if self.verbose and stop_count > 0:
            console.print(f"[dim]标记静止点: {stop_count} 个 (速度 < {STOP_SPEED_KMH} km/h 持续 > {STOP_DURATION_THRESHOLD.total_seconds() / 60:.0f} 分钟)[/dim]")
        return points

    def _extract_stay_points(self, points: List[TrackPoint]) -> List[StayPoint]:
        stays: List[StayPoint] = []
        if not points:
            return stays

        n = len(points)
        i = 0
        while i < n:
            if points[i].speed_kmh < STOP_SPEED_KMH:
                j = i
                while j < n and points[j].speed_kmh < STOP_SPEED_KMH:
                    j += 1
                if j > i:
                    duration = points[j - 1].time - points[i].time
                    if duration >= STAY_DURATION_THRESHOLD:
                        cluster = points[i:j]
                        avg_lat = sum(p.latitude for p in cluster) / len(cluster)
                        avg_lon = sum(p.longitude for p in cluster) / len(cluster)
                        stays.append(StayPoint(
                            location=(avg_lat, avg_lon),
                            start_time=points[i].time,
                            end_time=points[j - 1].time,
                        ))
                i = j
            else:
                i += 1

        if self.verbose and stays:
            console.print(f"[dim]识别停留/办事点: {len(stays)} 处[/dim]")
        return stays

    def _split_segments(self, points: List[TrackPoint]) -> List[TripSegment]:
        if not points:
            return []

        segments: List[TripSegment] = []
        current_points: List[TrackPoint] = []
        seg_id = 0

        def _finalize(pts, stays):
            nonlocal seg_id
            if pts:
                seg = TripSegment(points=pts, stay_points=stays, segment_id=seg_id)
                segments.append(seg)
                seg_id += 1

        i = 0
        n = len(points)

        while i < n:
            cur = points[i]

            if i == 0:
                current_points.append(cur)
                i += 1
                continue

            prev = points[i - 1]
            time_gap = cur.time - prev.time

            need_split = False

            if time_gap >= SEGMENT_GAP_THRESHOLD:
                need_split = True
                if self.verbose:
                    console.print(f"[dim]行程中断 (时间间隔 {time_gap.total_seconds()/60:.1f} 分钟): 在 {cur.time} 处分割[/dim]")
            else:
                was_moving = not prev.is_stop and prev.speed_kmh > STOP_SPEED_KMH
                now_stopped = cur.speed_kmh < STOP_SPEED_KMH

                if was_moving and now_stopped:
                    j = i
                    while j < n and points[j].speed_kmh < STOP_SPEED_KMH:
                        j += 1
                    if j > i:
                        stop_dur = points[j - 1].time - cur.time
                        if stop_dur >= STOP_TO_MOVE_THRESHOLD:
                            need_split = True
                            if self.verbose:
                                console.print(f"[dim]行程中断 (停车 {stop_dur.total_seconds()/60:.1f} 分钟): 在 {cur.time} 处分割[/dim]")

            if need_split:
                stays = self._extract_stay_points(current_points)
                _finalize(current_points, stays)
                current_points = []

            current_points.append(cur)
            i += 1

        stays = self._extract_stay_points(current_points)
        _finalize(current_points, stays)

        if self.verbose:
            console.print(f"[green]行程段分割完成: {len(segments)} 段[/green]")
        return segments

    def parse(self, file_path: str, employee_name: str = "") -> TripResult:
        if self.verbose:
            console.print(f"[bold blue]开始解析: {file_path}[/bold blue]")

        raw_points = self.parse_gpx(file_path)
        if not raw_points:
            raise ValueError("No valid track points found in GPX file")

        raw_points.sort(key=lambda p: p.time)
        points = self._compute_speeds_and_distances(raw_points)
        points = self._filter_drift_points(points)
        points = self._compute_speeds_and_distances(points)
        points = self._mark_stop_points(points)
        segments = self._split_segments(points)

        result = TripResult(
            file_path=file_path,
            segments=segments,
            employee_name=employee_name,
        )
        result.recompute_totals()

        if self.verbose:
            console.print(f"[bold green]解析完成: {len(segments)} 段行程, 总里程 {result.total_distance_km:.2f} km[/bold green]")
        return result

    def parse_directory(self, dir_path: str, employee_name: str = "") -> List[TripResult]:
        d = Path(dir_path)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        gpx_files = sorted(d.glob("**/*.gpx"))
        if not gpx_files:
            return []

        results: List[TripResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("批量解析GPX文件...", total=len(gpx_files))
            for f in gpx_files:
                try:
                    res = self.parse(str(f), employee_name=employee_name)
                    results.append(res)
                except Exception as e:
                    if self.verbose:
                        console.print(f"[red]解析失败 {f.name}: {e}[/red]")
                progress.advance(task)

        return results
