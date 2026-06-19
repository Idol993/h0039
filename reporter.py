from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import jinja2
import pandas as pd
import requests
from geopy.distance import geodesic
from rich.console import Console
from rich.table import Table

from trip_parser import StayPoint, TripResult, TripSegment

console = Console()

COMPLIANCE_DISTANCE_KM = 300.0
ROUTE_DEVIATION_THRESHOLD = 0.20

DEFAULT_OSRM_BASE = "https://router.project-osrm.org"


@dataclass
class RouteComparison:
    segment_id: int
    actual_distance_km: float
    recommended_distance_km: float
    deviation_percent: float
    is_detour: bool
    recommended_route_polyline: Optional[str] = None

    @property
    def status_label(self) -> str:
        if self.is_detour:
            return "⚠️ 疑似绕路"
        return "✓ 路线正常"


@dataclass
class SegmentReport:
    segment: TripSegment
    date: str
    start_coord_label: str
    end_coord_label: str
    distance_km: float
    duration: timedelta
    avg_speed_kmh: float
    route_check: Optional[RouteComparison] = None
    compliance_status: str = "合规"
    compliance_warning: bool = False


@dataclass
class FullReport:
    employee_name: str
    report_title: str
    generated_at: str
    segments: List[SegmentReport] = field(default_factory=list)
    stay_points: List[StayPoint] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_duration: timedelta = timedelta(0)
    total_moving_time: timedelta = timedelta(0)
    max_single_segment_km: float = 0.0
    compliance_violations: List[str] = field(default_factory=list)
    date_range: str = ""


def _format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}小时{minutes}分"
    if minutes > 0:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def _coord_label(coord: Optional[Tuple[float, float]]) -> str:
    if not coord:
        return "N/A"
    lat, lon = coord
    return f"({lat:.5f}, {lon:.5f})"


class RouteComparator:
    def __init__(self, osrm_base: str = DEFAULT_OSRM_BASE, timeout: int = 15):
        self.osrm_base = osrm_base.rstrip("/")
        self.timeout = timeout

    def _query_route(self, coords: List[Tuple[float, float]]) -> Optional[Dict]:
        if len(coords) < 2:
            return None
        coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
        url = f"{self.osrm_base}/route/v1/driving/{coord_str}"
        params = {
            "overview": "simplified",
            "geometries": "polyline",
            "steps": "false",
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    return data["routes"][0]
        except requests.RequestException as e:
            console.print(f"[yellow]OSRM请求失败: {e}[/yellow]")
        return None

    def compare_segment(self, segment: TripSegment) -> Optional[RouteComparison]:
        if len(segment.points) < 2:
            return None

        start = segment.points[0].coord
        end = segment.points[-1].coord

        waypoints = [start]
        if len(segment.points) >= 4:
            mid_idx = len(segment.points) // 2
            waypoints.append(segment.points[mid_idx].coord)
        waypoints.append(end)

        route_info = self._query_route(waypoints)
        if not route_info:
            return None

        recommended_distance_m = route_info.get("distance", 0.0)
        recommended_km = recommended_distance_m / 1000.0
        actual_km = segment.total_distance_km

        if recommended_km <= 0:
            return None

        deviation = (actual_km - recommended_km) / recommended_km
        is_detour = deviation > ROUTE_DEVIATION_THRESHOLD

        polyline_str = route_info.get("geometry")

        return RouteComparison(
            segment_id=segment.segment_id,
            actual_distance_km=actual_km,
            recommended_distance_km=recommended_km,
            deviation_percent=deviation * 100,
            is_detour=is_detour,
            recommended_route_polyline=polyline_str,
        )

    def compare_trip(self, result: TripResult) -> List[RouteComparison]:
        comparisons: List[RouteComparison] = []
        for seg in result.segments:
            comp = self.compare_segment(seg)
            if comp:
                comparisons.append(comp)
        return comparisons


class Reporter:
    def __init__(self, verbose: bool = False, osrm_base: str = DEFAULT_OSRM_BASE):
        self.verbose = verbose
        self.route_comparator = RouteComparator(osrm_base=osrm_base)
        self._template_env = None

    def _get_template_env(self) -> jinja2.Environment:
        if self._template_env is None:
            template_dir = Path(__file__).parent / "templates"
            if template_dir.exists():
                loader = jinja2.FileSystemLoader(str(template_dir))
            else:
                loader = jinja2.DictLoader(self._fallback_templates())
            self._template_env = jinja2.Environment(loader=loader, autoescape=False)
            self._template_env.filters["format_td"] = _format_timedelta
            self._template_env.filters["format_coord"] = _coord_label
        return self._template_env

    @staticmethod
    def _fallback_templates() -> Dict[str, str]:
        return {
            "report.md.j2": _FALLBACK_MD_TEMPLATE,
            "report.html.j2": _FALLBACK_HTML_TEMPLATE,
        }

    def _check_compliance(self, segment: TripSegment) -> Tuple[str, bool]:
        dist = segment.total_distance_km
        if dist > COMPLIANCE_DISTANCE_KM:
            return f"⚠️ 超过{COMPLIANCE_DISTANCE_KM:.0f}公里，需提前审批", True
        return "合规", False

    def build_segment_reports(
        self, trip: TripResult, check_route: bool = True
    ) -> List[SegmentReport]:
        route_comps: Dict[int, RouteComparison] = {}
        if check_route:
            if self.verbose:
                console.print("[dim]正在请求OSRM路线对比...[/dim]")
            comps = self.route_comparator.compare_trip(trip)
            route_comps = {c.segment_id: c for c in comps}

        seg_reports: List[SegmentReport] = []
        all_stays: List[StayPoint] = []

        for seg in trip.segments:
            date_str = seg.start_time.strftime("%Y-%m-%d") if seg.start_time else "N/A"
            status, warn = self._check_compliance(seg)
            r = SegmentReport(
                segment=seg,
                date=date_str,
                start_coord_label=_coord_label(seg.start_point),
                end_coord_label=_coord_label(seg.end_point),
                distance_km=seg.total_distance_km,
                duration=seg.duration,
                avg_speed_kmh=seg.avg_speed_kmh,
                route_check=route_comps.get(seg.segment_id),
                compliance_status=status,
                compliance_warning=warn,
            )
            seg_reports.append(r)
            all_stays.extend(seg.stay_points)

        return seg_reports, all_stays

    def build_full_report(
        self,
        trips: List[TripResult],
        employee_name: str = "",
        report_title: str = "员工出差行程报告",
        check_route: bool = True,
    ) -> FullReport:
        all_seg_reports: List[SegmentReport] = []
        all_stays: List[StayPoint] = []
        total_dist = 0.0
        max_single = 0.0
        violations: List[str] = []

        first_time: Optional[datetime] = None
        last_time: Optional[datetime] = None
        total_moving_seconds = 0.0

        for trip in trips:
            seg_reports, stays = self.build_segment_reports(trip, check_route=check_route)
            all_seg_reports.extend(seg_reports)
            all_stays.extend(stays)
            total_dist += trip.total_distance_km

            for seg in trip.segments:
                max_single = max(max_single, seg.total_distance_km)
                if seg.start_time:
                    first_time = min(first_time, seg.start_time) if first_time else seg.start_time
                if seg.end_time:
                    last_time = max(last_time, seg.end_time) if last_time else seg.end_time
                for p in seg.points:
                    if not p.is_stop and p.speed_kmh >= STOP_SPEED_DISPLAY:
                        time_s = 0
                        idx = seg.points.index(p)
                        if idx > 0:
                            prev = seg.points[idx - 1]
                            time_s = (p.time - prev.time).total_seconds()
                        total_moving_seconds += time_s

        for sr in all_seg_reports:
            if sr.compliance_warning:
                violations.append(
                    f"{sr.date} 第{sr.segment.segment_id + 1}段行程 {sr.distance_km:.1f}km 超出审批标准"
                )

        name = employee_name or (trips[0].employee_name if trips else "未知员工")
        date_range = ""
        if first_time and last_time:
            date_range = f"{first_time.strftime('%Y-%m-%d')} ~ {last_time.strftime('%Y-%m-%d')}"

        total_duration = (last_time - first_time) if (first_time and last_time) else timedelta(0)
        total_moving = timedelta(seconds=int(total_moving_seconds))

        return FullReport(
            employee_name=name,
            report_title=report_title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            segments=all_seg_reports,
            stay_points=all_stays,
            total_distance_km=total_dist,
            total_duration=total_duration,
            total_moving_time=total_moving,
            max_single_segment_km=max_single,
            compliance_violations=violations,
            date_range=date_range,
        )

    def render_markdown(self, report: FullReport) -> str:
        env = self._get_template_env()
        template = env.get_template("report.md.j2")
        return template.render(report=report, COMPLIANCE_LIMIT=COMPLIANCE_DISTANCE_KM)

    def render_html(self, report: FullReport) -> str:
        env = self._get_template_env()
        template = env.get_template("report.html.j2")
        return template.render(
            report=report,
            COMPLIANCE_LIMIT=COMPLIANCE_DISTANCE_KM,
            DEVIATION_THRESHOLD=ROUTE_DEVIATION_THRESHOLD * 100,
        )

    def save_markdown(self, report: FullReport, output_path: str) -> str:
        md = self.render_markdown(report)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        if self.verbose:
            console.print(f"[green]Markdown报告已保存: {p}[/green]")
        return str(p)

    def save_html(self, report: FullReport, output_path: str) -> str:
        html = self.render_html(report)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        if self.verbose:
            console.print(f"[green]HTML报告已保存: {p}[/green]")
        return str(p)

    def export_csv(self, report: FullReport, output_path: str) -> str:
        rows = []
        for idx, sr in enumerate(report.segments):
            seg = sr.segment
            rows.append({
                "序号": idx + 1,
                "日期": sr.date,
                "起点经度": seg.start_point[1] if seg.start_point else "",
                "起点纬度": seg.start_point[0] if seg.start_point else "",
                "终点经度": seg.end_point[1] if seg.end_point else "",
                "终点纬度": seg.end_point[0] if seg.end_point else "",
                "里程(km)": f"{sr.distance_km:.2f}",
                "开始时间": seg.start_time.strftime("%Y-%m-%d %H:%M:%S") if seg.start_time else "",
                "结束时间": seg.end_time.strftime("%Y-%m-%d %H:%M:%S") if seg.end_time else "",
                "时长": _format_timedelta(sr.duration),
                "平均速度(km/h)": f"{sr.avg_speed_kmh:.1f}",
                "合规状态": sr.compliance_status,
                "最短路线(km)": f"{sr.route_check.recommended_distance_km:.2f}" if sr.route_check else "",
                "偏差(%)": f"{sr.route_check.deviation_percent:.1f}" if sr.route_check else "",
                "路线状态": sr.route_check.status_label if sr.route_check else "",
                "员工": report.employee_name,
            })
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(p, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        if self.verbose:
            console.print(f"[green]CSV已导出: {p}[/green]")
        return str(p)

    def export_points_csv(self, trips: List[TripResult], output_path: str) -> str:
        rows = []
        seq = 0
        for trip in trips:
            fname = Path(trip.file_path).name
            for seg in trip.segments:
                for p in seg.points:
                    seq += 1
                    rows.append({
                        "id": seq,
                        "file": fname,
                        "segment_id": seg.segment_id,
                        "time": p.time.isoformat() if p.time else "",
                        "latitude": f"{p.latitude:.7f}",
                        "longitude": f"{p.longitude:.7f}",
                        "elevation_m": f"{p.elevation:.1f}" if p.elevation is not None else "",
                        "speed_kmh": f"{p.speed_kmh:.2f}",
                        "distance_m": f"{p.distance_m:.2f}",
                        "is_stop": "1" if p.is_stop else "0",
                        "is_drift": "1" if p.is_drift else "0",
                        "employee": trip.employee_name,
                    })
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        if self.verbose:
            console.print(f"[green]轨迹点CSV已导出: {p} ({len(rows)} 行)[/green]")
        return str(p)

    def generate_map(self, trips: List[TripResult], output_path: str, report: Optional[FullReport] = None) -> str:
        all_points = []
        for trip in trips:
            for seg in trip.segments:
                all_points.extend([p.coord for p in seg.points])

        if not all_points:
            raise ValueError("No points to render map")

        center_lat = sum(p[0] for p in all_points) / len(all_points)
        center_lon = sum(p[1] for p in all_points) / len(all_points)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

        colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#34495e", "#d35400", "#27ae60",
        ]

        route_map: Dict[int, RouteComparison] = {}
        if report:
            for sr in report.segments:
                if sr.route_check:
                    route_map[sr.segment.segment_id] = sr.route_check

        seg_global_idx = 0
        for trip in trips:
            for seg in trip.segments:
                color = colors[seg_global_idx % len(colors)]
                seg_global_idx += 1

                if not seg.points:
                    continue

                latlons = [[p.latitude, p.longitude] for p in seg.points]
                folium.PolyLine(
                    locations=latlons,
                    color=color,
                    weight=4,
                    opacity=0.85,
                    tooltip=f"第{seg.segment_id + 1}段 · {seg.total_distance_km:.1f}km · {_format_timedelta(seg.duration)}",
                ).add_to(m)

                sp = seg.points[0]
                folium.Marker(
                    location=[sp.latitude, sp.longitude],
                    icon=folium.Icon(color="green", icon="play"),
                    popup=f"起点<br>时间: {sp.time.strftime('%H:%M:%S')}<br>第{seg.segment_id + 1}段",
                ).add_to(m)

                ep = seg.points[-1]
                folium.Marker(
                    location=[ep.latitude, ep.longitude],
                    icon=folium.Icon(color="red", icon="flag"),
                    popup=f"终点<br>时间: {ep.time.strftime('%H:%M:%S')}<br>里程: {seg.total_distance_km:.2f}km",
                ).add_to(m)

                rc = route_map.get(seg.segment_id)
                if rc and rc.recommended_route_polyline:
                    try:
                        import polyline as polyline_lib
                        decoded = polyline_lib.decode(rc.recommended_route_polyline)
                        if decoded:
                            rec_latlons = [[lat, lon] for lat, lon in decoded]
                            status_color = "#f1c40f" if rc.is_detour else "#95a5a6"
                            folium.PolyLine(
                                locations=rec_latlons,
                                color=status_color,
                                weight=3,
                                opacity=0.6,
                                dash_array="8, 8",
                                tooltip=f"推荐路线 {rc.recommended_distance_km:.1f}km · 偏差{rc.deviation_percent:.1f}%",
                            ).add_to(m)
                    except Exception:
                        pass

                for stay in seg.stay_points:
                    folium.CircleMarker(
                        location=[stay.center_lat, stay.center_lon],
                        radius=10,
                        color="#3498db",
                        fill=True,
                        fill_color="#3498db",
                        fill_opacity=0.5,
                        popup=f"停留/办事<br>时长: {_format_timedelta(stay.duration)}<br>{stay.start_time.strftime('%H:%M')} - {stay.end_time.strftime('%H:%M')}",
                    ).add_to(m)

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 12px;
            left: 12px;
            z-index: 9999;
            background-color: rgba(255,255,255,0.95);
            padding: 10px 14px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        ">
            <div style="font-weight:bold;margin-bottom:6px;">🗺️ 轨迹图说明</div>
            <div>🟢 起点 / 🔴 终点 / 🔵 停留点</div>
            <div style="margin-top:4px;">实线 = 实际轨迹，虚线 = 推荐路线</div>
            <div style="margin-top:4px;color:#e67e22;">黄色虚线 = 偏差>{ROUTE_DEVIATION_THRESHOLD*100:.0f}%（疑似绕路）</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(p))
        if self.verbose:
            console.print(f"[green]交互式地图已保存: {p}[/green]")
        return str(p)

    def save_pdf(self, report: FullReport, output_path: str) -> Optional[str]:
        import shutil
        import subprocess
        import tempfile

        html_content = self.render_html(report)
        wkhtmltopdf = shutil.which("wkhtmltopdf")

        if wkhtmltopdf:
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tf:
                tf.write(html_content)
                tf_path = tf.name
            try:
                p = Path(output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [wkhtmltopdf, "--enable-local-file-access", tf_path, str(p)],
                    check=True,
                    capture_output=True,
                )
                if self.verbose:
                    console.print(f"[green]PDF已保存 (wkhtmltopdf): {p}[/green]")
                return str(p)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if self.verbose:
                    console.print(f"[yellow]wkhtmltopdf失败: {e}，尝试markdown-pdf方式...[/yellow]")
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass
        try:
            from markdown import markdown as md_convert
            md = self.render_markdown(report)
            html_body = md_convert(md, extensions=["tables", "fenced_code"])
            html_full = f"<html><head><meta charset='utf-8'><style>body{{font-family:Microsoft YaHei,Arial;padding:24px;}}table{{border-collapse:collapse;width:100%;}}th,td{{border:1px solid #ccc;padding:8px;text-align:left;}}th{{background:#f0f0f0;}}.warn{{color:#e67e22;}}.danger{{color:#c0392b;font-weight:bold;}}</style></head><body>{html_body}</body></html>"
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tf:
                tf.write(html_full)
                tf_path = tf.name
            try:
                mdpdf = shutil.which("mdpdf") or shutil.which("markdown-pdf")
                if mdpdf:
                    subprocess.run([mdpdf, tf_path, "-o", output_path], check=True)
                else:
                    p = Path(output_path)
                    html_path = p.with_suffix(".html")
                    html_path.write_text(html_full, encoding="utf-8")
                    console.print(f"[yellow]未找到PDF转换工具，已保存HTML: {html_path}[/yellow]")
                    return str(html_path)
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass
        except Exception as e:
            if self.verbose:
                console.print(f"[red]PDF导出失败: {e}[/red]")
        return None

    def print_summary(self, report: FullReport) -> None:
        console.print()
        title = f"📋 {report.report_title}"
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]{title.center(54)}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"员工姓名: [bold]{report.employee_name}[/bold]")
        if report.date_range:
            console.print(f"日期范围: {report.date_range}")
        console.print(f"生成时间: {report.generated_at}")
        console.print()

        tbl = Table(show_header=True, header_style="bold magenta")
        tbl.add_column("序号", justify="center")
        tbl.add_column("日期")
        tbl.add_column("起点")
        tbl.add_column("终点")
        tbl.add_column("里程(km)", justify="right")
        tbl.add_column("时长", justify="right")
        tbl.add_column("均速(km/h)", justify="right")
        tbl.add_column("路线")
        tbl.add_column("合规")

        for idx, sr in enumerate(report.segments, 1):
            route_txt = sr.route_check.status_label if sr.route_check else "-"
            route_style = "yellow" if (sr.route_check and sr.route_check.is_detour) else "green"
            comp_style = "bold red" if sr.compliance_warning else "green"

            tbl.add_row(
                str(idx),
                sr.date,
                sr.start_coord_label,
                sr.end_coord_label,
                f"{sr.distance_km:.2f}",
                _format_timedelta(sr.duration),
                f"{sr.avg_speed_kmh:.1f}",
                f"[{route_style}]{route_txt}[/{route_style}]" if sr.route_check else route_txt,
                f"[{comp_style}]{sr.compliance_status}[/{comp_style}]",
            )
        console.print(tbl)

        console.print()
        stats = Table(show_header=False, box=None)
        stats.add_column("项目", style="bold")
        stats.add_column("数值", justify="right")
        stats.add_row("行程段数", f"{len(report.segments)} 段")
        stats.add_row("停留/办事次数", f"{len(report.stay_points)} 次")
        stats.add_row("总里程", f"[bold]{report.total_distance_km:.2f}[/bold] km")
        stats.add_row("总时长", _format_timedelta(report.total_duration))
        stats.add_row("行驶时长", _format_timedelta(report.total_moving_time))
        stats.add_row("最大单段里程", f"{report.max_single_segment_km:.2f} km")
        stats.add_row("合规状态", "[bold red]有超标行程[/bold red]" if report.compliance_violations else "[bold green]全部合规[/bold green]")
        console.print(stats)

        if report.compliance_violations:
            console.print()
            console.print("[bold red]⚠️ 合规警告:[/bold red]")
            for v in report.compliance_violations:
                console.print(f"  [red]• {v}[/red]")
        console.print()


STOP_SPEED_DISPLAY = 1.0


_FALLBACK_MD_TEMPLATE = """# {{ report.report_title }}

**员工姓名**: {{ report.employee_name }}
{% if report.date_range %}
**日期范围**: {{ report.date_range }}
{% endif %}
**生成时间**: {{ report.generated_at }}

---

## 📊 行程总览

| 序号 | 日期 | 起点 | 终点 | 里程(km) | 时长 | 平均速度(km/h) | 推荐路线(km) | 偏差(%) | 路线 | 合规 |
|------|------|------|------|----------|------|----------------|--------------|---------|------|------|
{% for s in report.segments %}| {{ loop.index }} | {{ s.date }} | {{ s.start_coord_label }} | {{ s.end_coord_label }} | {{ "%.2f"|format(s.distance_km) }} | {{ s.duration | format_td }} | {{ "%.1f"|format(s.avg_speed_kmh) }} | {% if s.route_check %}{{ "%.2f"|format(s.route_check.recommended_distance_km) }}{% else %}-{% endif %} | {% if s.route_check %}{{ "%.1f"|format(s.route_check.deviation_percent) }}{% else %}-{% endif %} | {% if s.route_check %}{{ s.route_check.status_label }}{% else %}-{% endif %} | {{ s.compliance_status }} |
{% endfor %}

---

## 📈 统计汇总

- **行程段数**: {{ report.segments|length }} 段
- **停留/办事次数**: {{ report.stay_points|length }} 次
- **总里程**: **{{ "%.2f"|format(report.total_distance_km) }} km**
- **总时长**: {{ report.total_duration | format_td }}
- **行驶时长**: {{ report.total_moving_time | format_td }}
- **最大单段里程**: {{ "%.2f"|format(report.max_single_segment_km) }} km
- **审批阈值**: 单段 > {{ COMPLIANCE_LIMIT }} km 需提前审批

{% if report.compliance_violations %}
## ⚠️ 合规警告

{% for v in report.compliance_violations %}
- {{ v }}
{% endfor %}
{% endif %}

---

## 🔍 各段行程详情

{% for s in report.segments %}
### 第 {{ loop.index }} 段 · {{ s.date }}

- **起点**: {{ s.start_coord_label }}
- **终点**: {{ s.end_coord_label }}
- **起止时间**: {{ s.segment.start_time.strftime('%H:%M:%S') if s.segment.start_time else 'N/A' }} ~ {{ s.segment.end_time.strftime('%H:%M:%S') if s.segment.end_time else 'N/A' }}
- **实际里程**: {{ "%.2f"|format(s.distance_km) }} km
- **行驶时长**: {{ s.duration | format_td }}
- **平均速度**: {{ "%.1f"|format(s.avg_speed_kmh) }} km/h
{% if s.route_check %}
- **推荐最短路线**: {{ "%.2f"|format(s.route_check.recommended_distance_km) }} km
- **路线偏差**: {{ "%.1f"|format(s.route_check.deviation_percent) }} % {{ '⚠️ 疑似绕路' if s.route_check.is_detour else '✓ 正常' }}
{% endif %}
- **合规状态**: {{ s.compliance_status }}

{% if s.segment.stay_points %}
#### 停留/办事点

| # | 位置 | 到达 | 离开 | 停留时长 |
|---|------|------|------|----------|
{% for stay in s.segment.stay_points %}| {{ loop.index }} | ({{ "%.5f"|format(stay.center_lat) }}, {{ "%.5f"|format(stay.center_lon) }}) | {{ stay.start_time.strftime('%H:%M') }} | {{ stay.end_time.strftime('%H:%M') }} | {{ stay.duration | format_td }} |
{% endfor %}
{% endif %}

---
{% endfor %}

*报告由 TripLog 自动生成，GPS数据可能存在误差，仅供参考。*
"""


_FALLBACK_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ report.report_title }} - {{ report.employee_name }}</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", "PingFang SC", sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 28px 32px;
    color: #2c3e50;
    line-height: 1.6;
    background: #f7f9fc;
  }
  h1 { color: #2980b9; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #27ae60; margin-top: 28px; border-left: 5px solid #27ae60; padding-left: 10px; }
  h3 { color: #8e44ad; }
  .header-info {
    background: #fff;
    padding: 16px 24px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
  }
  .header-info div { padding: 4px 0; }
  .header-info b { color: #34495e; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    background: #fff;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
    font-size: 14px;
  }
  th { background: linear-gradient(to bottom, #3498db, #2980b9); color: #fff; font-weight: 600; }
  tr:hover td { background: #f8fbff; }
  tr:last-child td { border-bottom: none; }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 14px 0;
  }
  .stat-card {
    background: #fff;
    padding: 14px 18px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    border-top: 3px solid #3498db;
  }
  .stat-card .label { font-size: 13px; color: #7f8c8d; }
  .stat-card .value { font-size: 22px; font-weight: 700; color: #2c3e50; margin-top: 4px; }
  .warn { background: #fff8e1 !important; color: #e67e22; font-weight: 600; }
  .danger { background: #ffebee !important; color: #c0392b; font-weight: 700; }
  .ok { color: #27ae60; font-weight: 600; }
  .seg-box {
    background: #fff;
    padding: 18px 22px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    margin: 14px 0;
    border-left: 4px solid #3498db;
  }
  .seg-meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin: 10px 0;
  }
  .seg-meta div { padding: 6px 10px; background: #f4f7fa; border-radius: 4px; }
  .seg-meta span { font-size: 12px; color: #7f8c8d; display: block; }
  .seg-meta b { color: #2c3e50; }
  .alert {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 12px 18px;
    border-radius: 6px;
    margin: 14px 0;
  }
  .alert-danger { background: #f8d7da; border-color: #f5c6cb; color: #721c24; }
  .footer { margin-top: 40px; text-align: center; color: #95a5a6; font-size: 12px; padding: 18px; border-top: 1px solid #ecf0f1; }
</style>
</head>
<body>

<h1>📋 {{ report.report_title }}</h1>

<div class="header-info">
  <div><b>员工姓名:</b> {{ report.employee_name }}</div>
  {% if report.date_range %}<div><b>日期范围:</b> {{ report.date_range }}</div>{% endif %}
  <div><b>生成时间:</b> {{ report.generated_at }}</div>
  <div><b>审批阈值:</b> 单段 > {{ COMPLIANCE_LIMIT }} km</div>
</div>

<h2>📊 行程总览表</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>日期</th><th>起点</th><th>终点</th><th>里程(km)</th>
      <th>时长</th><th>均速</th><th>推荐路线</th><th>偏差</th><th>路线</th><th>合规</th>
    </tr>
  </thead>
  <tbody>
  {% for s in report.segments %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ s.date }}</td>
      <td>{{ s.start_coord_label }}</td>
      <td>{{ s.end_coord_label }}</td>
      <td><b>{{ "%.2f"|format(s.distance_km) }}</b></td>
      <td>{{ s.duration | format_td }}</td>
      <td>{{ "%.1f"|format(s.avg_speed_kmh) }} km/h</td>
      <td>{% if s.route_check %}{{ "%.2f"|format(s.route_check.recommended_distance_km) }} km{% else %}-{% endif %}</td>
      <td class="{% if s.route_check and s.route_check.deviation_percent > DEVIATION_THRESHOLD %}warn{% endif %}">{% if s.route_check %}{{ "%.1f"|format(s.route_check.deviation_percent) }}%{% else %}-{% endif %}</td>
      <td class="{% if s.route_check and s.route_check.is_detour %}warn{% else %}ok{% endif %}">{% if s.route_check %}{{ s.route_check.status_label }}{% else %}-{% endif %}</td>
      <td class="{% if s.compliance_warning %}danger{% else %}ok{% endif %}">{{ s.compliance_status }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2>📈 统计汇总</h2>
<div class="stats-grid">
  <div class="stat-card"><div class="label">行程段数</div><div class="value">{{ report.segments|length }} 段</div></div>
  <div class="stat-card"><div class="label">停留/办事次数</div><div class="value">{{ report.stay_points|length }} 次</div></div>
  <div class="stat-card" style="border-top-color:#e74c3c;"><div class="label">总里程</div><div class="value" style="color:#c0392b;">{{ "%.2f"|format(report.total_distance_km) }} km</div></div>
  <div class="stat-card"><div class="label">总时长</div><div class="value">{{ report.total_duration | format_td }}</div></div>
  <div class="stat-card"><div class="label">行驶时长</div><div class="value">{{ report.total_moving_time | format_td }}</div></div>
  <div class="stat-card"><div class="label">最大单段里程</div><div class="value">{{ "%.2f"|format(report.max_single_segment_km) }} km</div></div>
</div>

{% if report.compliance_violations %}
<div class="alert alert-danger">
  <b>⚠️ 合规警告 ({{ report.compliance_violations|length }} 项):</b>
  <ul style="margin:8px 0 0 20px;">
  {% for v in report.compliance_violations %}
    <li>{{ v }}</li>
  {% endfor %}
  </ul>
</div>
{% endif %}

<h2>🔍 各段行程详情</h2>

{% for s in report.segments %}
<div class="seg-box">
  <h3>🚗 第 {{ loop.index }} 段 · {{ s.date }}</h3>
  <div class="seg-meta">
    <div><span>起点坐标</span><b>{{ s.start_coord_label }}</b></div>
    <div><span>终点坐标</span><b>{{ s.end_coord_label }}</b></div>
    <div><span>起止时间</span><b>{{ s.segment.start_time.strftime('%H:%M:%S') if s.segment.start_time else 'N/A' }} ~ {{ s.segment.end_time.strftime('%H:%M:%S') if s.segment.end_time else 'N/A' }}</b></div>
    <div><span>实际里程</span><b>{{ "%.2f"|format(s.distance_km) }} km</b></div>
    <div><span>行驶时长</span><b>{{ s.duration | format_td }}</b></div>
    <div><span>平均速度</span><b>{{ "%.1f"|format(s.avg_speed_kmh) }} km/h</b></div>
    {% if s.route_check %}
    <div><span>推荐最短路线</span><b>{{ "%.2f"|format(s.route_check.recommended_distance_km) }} km</b></div>
    <div class="{% if s.route_check.is_detour %}alert{% endif %}" style="{% if s.route_check.is_detour %}padding:6px 10px;{% endif %}"><span>路线偏差</span><b>{{ "%.1f"|format(s.route_check.deviation_percent) }} % {{ '⚠️ 疑似绕路' if s.route_check.is_detour else '✓ 正常' }}</b></div>
    {% endif %}
    <div class="{% if s.compliance_warning %}alert-danger alert{% endif %}" style="{% if s.compliance_warning %}padding:6px 10px;{% endif %}"><span>合规状态</span><b>{{ s.compliance_status }}</b></div>
  </div>

  {% if s.segment.stay_points %}
  <h4 style="margin-top:14px;color:#3498db;">⏸️ 停留/办事点</h4>
  <table>
    <thead><tr><th>#</th><th>位置坐标</th><th>到达时间</th><th>离开时间</th><th>停留时长</th></tr></thead>
    <tbody>
    {% for stay in s.segment.stay_points %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>({{ "%.5f"|format(stay.center_lat) }}, {{ "%.5f"|format(stay.center_lon) }})</td>
        <td>{{ stay.start_time.strftime('%H:%M') }}</td>
        <td>{{ stay.end_time.strftime('%H:%M') }}</td>
        <td><b>{{ stay.duration | format_td }}</b></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endfor %}

<div class="footer">
  本报告由 TripLog 行程管理工具自动生成 · 数据仅供报销参考 · GPS轨迹数据可能存在 ±{{ DEVIATION_THRESHOLD }}% 误差
</div>
</body>
</html>
"""
