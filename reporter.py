from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
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
from policy import PolicyEngine, SegmentReimbursement

console = Console()

COMPLIANCE_DISTANCE_KM = 300.0
ROUTE_DEVIATION_THRESHOLD = 0.20
OSRM_MAX_RETRIES = 3
OSRM_RETRY_DELAY = 1.5
OSRM_CACHE_TTL = 86400 * 30

DEFAULT_OSRM_BASE = "https://router.project-osrm.org"

CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "南京": (32.0603, 118.7969),
    "成都": (30.5728, 104.0668),
    "天津": (39.3434, 117.3616),
    "重庆": (29.5630, 106.5516),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "苏州": (31.2990, 120.5853),
    "无锡": (31.4912, 120.3119),
    "青岛": (36.0671, 120.3826),
    "厦门": (24.4798, 118.0894),
    "福州": (26.0745, 119.2965),
    "长沙": (28.2282, 112.9388),
    "郑州": (34.7466, 113.6254),
    "合肥": (31.8206, 117.2272),
    "济南": (36.6512, 117.1201),
}
CITY_DETECTION_RADIUS_KM = 80.0


def detect_city(
    start_coord: Optional[Tuple[float, float]],
    end_coord: Optional[Tuple[float, float]],
) -> str:
    """根据起终点坐标判断所属城市。起终点任意一方命中且距离 < CITY_DETECTION_RADIUS_KM 即认为属于该城市，优先起点城市。"""
    if not start_coord and not end_coord:
        return ""
    candidates: List[Tuple[str, float]] = []
    for city_name, (clat, clon) in CITY_COORDS.items():
        min_dist = float("inf")
        for pt in (start_coord, end_coord):
            if pt is None:
                continue
            try:
                d = geodesic((pt[0], pt[1]), (clat, clon)).km
                if d < min_dist:
                    min_dist = d
            except Exception:
                continue
        if min_dist < CITY_DETECTION_RADIUS_KM:
            candidates.append((city_name, min_dist))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def make_trip_source_id(file_path: str) -> str:
    """生成唯一的 trip_source_id，基于完整路径的 hash，避免跨目录同名文件冲突。"""
    try:
        full = Path(file_path).resolve(strict=False).as_posix()
    except Exception:
        full = str(file_path)
    h = hashlib.md5(full.encode("utf-8")).hexdigest()[:12]
    stem = Path(file_path).stem
    return f"{h}_{stem}"


@dataclass
class RouteComparison:
    segment_id: int
    actual_distance_km: float
    recommended_distance_km: float
    deviation_percent: float
    is_detour: bool
    recommended_route_polyline: Optional[str] = None
    verified: bool = True
    unverified_reason: str = ""  # manual(手动关闭) / network(网络失败) / no_result(服务无结果)

    @property
    def status_label(self) -> str:
        if not self.verified:
            if self.unverified_reason == "manual":
                return "⚠️ 未核验(手动关闭)"
            if self.unverified_reason == "no_result":
                return "⚠️ 未核验(路线服务无结果)"
            return "⚠️ 未核验(网络不可用)"
        if self.is_detour:
            return "⚠️ 疑似绕路"
        return "✓ 路线正常"

    @property
    def unverified_reason_display(self) -> str:
        if self.verified:
            return ""
        mapping = {
            "manual": "手动关闭路线核验",
            "network": "网络请求失败",
            "no_result": "路线服务未返回结果",
        }
        return mapping.get(self.unverified_reason, "未核验")


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
    reimbursement: Optional[SegmentReimbursement] = None
    policy_name: str = "default"
    trip_source_id: str = ""  # 用于地图推荐路线匹配，避免跨文件段号冲突

    @property
    def rec_distance_display(self) -> str:
        if not self.route_check or not self.route_check.verified:
            return "-"
        return f"{self.route_check.recommended_distance_km:.2f}"

    @property
    def deviation_display(self) -> str:
        if not self.route_check or not self.route_check.verified:
            return "-"
        return f"{self.route_check.deviation_percent:.1f}"

    @property
    def route_status_display(self) -> str:
        if not self.route_check:
            return "-"
        return self.route_check.status_label

    @property
    def is_unverified(self) -> bool:
        return not (self.route_check and self.route_check.verified)

    @property
    def unverified_reason_display(self) -> str:
        if self.route_check:
            return self.route_check.unverified_reason_display
        return "手动关闭路线核验"


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
    total_reimbursement_amount: float = 0.0
    approval_required: List[str] = field(default_factory=list)
    policy_name: str = "default"
    price_per_km: float = 1.5
    approval_threshold_km: Optional[float] = None
    # 审批流三类清单
    direct_reimburse: List[str] = field(default_factory=list)  # 可直接报销
    supervisor_approval: List[str] = field(default_factory=list)  # 待主管审批
    finance_review: List[str] = field(default_factory=list)  # 待财务复核
    # 未核验统计
    unverified_segments: List[str] = field(default_factory=list)  # 未核验段列表
    unverified_count: int = 0
    unverified_breakdown: Dict[str, int] = field(default_factory=dict)
    # 多政策汇总
    policy_breakdown: Dict[str, Dict] = field(default_factory=dict)  # {政策名: {段数, 里程, 金额, 单价}}


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
    def __init__(self, osrm_base: str = DEFAULT_OSRM_BASE, timeout: int = 15,
                 cache_dir: Optional[str] = None):
        self.osrm_base = osrm_base.rstrip("/")
        self.timeout = timeout
        self.cache_dir: Optional[Path] = None
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Dict] = {}

    def _cache_key(self, coords: List[Tuple[float, float]]) -> str:
        flat = "|".join(f"{lat:.5f},{lon:.5f}" for lat, lon in coords)
        return hashlib.sha1((self.osrm_base + "|" + flat).encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict]:
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry["ts"] < OSRM_CACHE_TTL:
                return entry["data"]
        if self.cache_dir is not None:
            fp = self.cache_dir / f"osrm_{key}.json"
            if fp.exists():
                try:
                    entry = json.loads(fp.read_text(encoding="utf-8"))
                    if time.time() - entry["ts"] < OSRM_CACHE_TTL:
                        self._memory_cache[key] = entry
                        return entry["data"]
                except Exception:
                    pass
        return None

    def _cache_put(self, key: str, data: Dict) -> None:
        entry = {"ts": time.time(), "data": data}
        self._memory_cache[key] = entry
        if self.cache_dir is not None:
            try:
                (self.cache_dir / f"osrm_{key}.json").write_text(
                    json.dumps(entry, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                pass

    def _query_route(self, coords: List[Tuple[float, float]]) -> Tuple[Optional[Dict], str]:
        if len(coords) < 2:
            return None, "no_result"
        key = self._cache_key(coords)
        cached = self._cache_get(key)
        if cached is not None:
            return cached, "ok"

        coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
        url = f"{self.osrm_base}/route/v1/driving/{coord_str}"
        params = {
            "overview": "simplified",
            "geometries": "polyline",
            "steps": "false",
        }
        last_exc: Optional[Exception] = None
        last_status: str = "no_result"
        for attempt in range(1, OSRM_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == "Ok" and data.get("routes"):
                        result = data["routes"][0]
                        self._cache_put(key, result)
                        return result, "ok"
                    else:
                        last_status = "no_result"
                elif resp.status_code == 429 or 500 <= resp.status_code < 600:
                    last_status = "network"
                    raise RuntimeError(f"HTTP {resp.status_code}")
                else:
                    last_status = "no_result"
            except requests.Timeout as e:
                last_exc = e
                last_status = "network"
            except requests.RequestException as e:
                last_exc = e
                last_status = "network"
            except Exception as e:
                last_exc = e
                # 非 requests 异常如果是我们主动抛的 RuntimeError(HTTP 429/5xx)，保持 network
                if last_status != "network":
                    last_status = "network"
            if attempt < OSRM_MAX_RETRIES:
                time.sleep(OSRM_RETRY_DELAY * attempt)

        if last_exc:
            console.print(f"[yellow]OSRM请求失败(已重试{OSRM_MAX_RETRIES}次): {last_exc}[/yellow]")
        return None, last_status

    def compare_segment(self, segment: TripSegment) -> RouteComparison:
        actual_km = segment.moving_distance_km if segment.moving_distance_km > 0 else segment.total_distance_km

        def _fallback(reason: str) -> RouteComparison:
            return RouteComparison(
                segment_id=segment.segment_id,
                actual_distance_km=actual_km,
                recommended_distance_km=0.0,
                deviation_percent=0.0,
                is_detour=False,
                recommended_route_polyline=None,
                verified=False,
                unverified_reason=reason,
            )

        if len(segment.points) < 2:
            return _fallback("no_result")

        start = segment.points[0].coord
        end = segment.points[-1].coord

        route_info, status = self._query_route([start, end])
        if not route_info or status != "ok":
            return _fallback(status if status in ("network", "no_result") else "no_result")

        recommended_distance_m = route_info.get("distance", 0.0)
        recommended_km = recommended_distance_m / 1000.0

        if recommended_km <= 0:
            return _fallback("no_result")

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
            verified=True,
            unverified_reason="",
        )

    def compare_trip(self, result: TripResult) -> List[RouteComparison]:
        comparisons: List[RouteComparison] = []
        for seg in result.segments:
            comparisons.append(self.compare_segment(seg))
        return comparisons


class Reporter:
    def __init__(self, verbose: bool = False, osrm_base: str = DEFAULT_OSRM_BASE,
                 cache_dir: Optional[str] = None, policy_config: Optional[str] = None,
                 employee_resolver=None):
        self.verbose = verbose
        self.route_comparator = RouteComparator(osrm_base=osrm_base, cache_dir=cache_dir)
        self.policy_engine = PolicyEngine(config_path=policy_config, verbose=verbose)
        self.employee_resolver = employee_resolver
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

    def _check_compliance(self, segment: TripSegment, threshold_km: float = COMPLIANCE_DISTANCE_KM) -> Tuple[str, bool]:
        dist = segment.total_distance_km
        if dist > threshold_km:
            return f"⚠️ 超过{threshold_km:.0f}公里，需提前审批", True
        return "合规", False

    def build_segment_reports(
        self, trip: TripResult, check_route: bool = True, employee_name: str = "",
        department: str = "",
    ) -> Tuple[List[SegmentReport], List[StayPoint]]:
        route_comps: Dict[int, RouteComparison] = {}
        if check_route:
            if self.verbose:
                console.print("[dim]正在请求OSRM路线对比...[/dim]")
            comps = self.route_comparator.compare_trip(trip)
            route_comps = {c.segment_id: c for c in comps}

        emp = employee_name or trip.employee_name
        seg_reports: List[SegmentReport] = []
        all_stays: List[StayPoint] = []
        trip_source = make_trip_source_id(trip.file_path) if trip.file_path else f"trip_{id(trip)}"

        for seg in trip.segments:
            date_str = seg.start_time.strftime("%Y-%m-%d") if seg.start_time else "N/A"
            city = detect_city(seg.start_point, seg.end_point)
            threshold = COMPLIANCE_DISTANCE_KM
            reimbursement = self.policy_engine.compute_reimbursement(
                seg, emp, date_str, city=city, department=department,
            )
            if reimbursement:
                matched_pol = self.policy_engine.find_policy(
                    emp, date_str, city=city, department=department,
                )
                threshold = matched_pol.approval_threshold_km

            status, warn = self._check_compliance(seg, threshold_km=threshold)
            moving_dur = seg.moving_time
            avg_speed = seg.avg_speed_kmh

            rc = route_comps.get(seg.segment_id)
            if not check_route and rc is None:
                actual_km = seg.moving_distance_km if seg.moving_distance_km > 0 else seg.total_distance_km
                rc = RouteComparison(
                    segment_id=seg.segment_id,
                    actual_distance_km=actual_km,
                    recommended_distance_km=0.0,
                    deviation_percent=0.0,
                    is_detour=False,
                    recommended_route_polyline=None,
                    verified=False,
                    unverified_reason="manual",
                )

            r = SegmentReport(
                segment=seg,
                date=date_str,
                start_coord_label=_coord_label(seg.start_point),
                end_coord_label=_coord_label(seg.end_point),
                distance_km=seg.moving_distance_km if seg.moving_distance_km > 0 else seg.total_distance_km,
                duration=moving_dur,
                avg_speed_kmh=avg_speed,
                route_check=rc,
                compliance_status=status,
                compliance_warning=warn,
                reimbursement=reimbursement,
                policy_name=reimbursement.policy_name if reimbursement else "default",
                trip_source_id=trip_source,
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
        department: str = "",
    ) -> FullReport:
        all_seg_reports: List[SegmentReport] = []
        all_stays: List[StayPoint] = []
        total_dist = 0.0
        max_single = 0.0
        violations: List[str] = []

        first_time: Optional[datetime] = None
        last_time: Optional[datetime] = None
        total_moving = timedelta(0)
        total_reimburse = 0.0
        approval_list: List[str] = []
        policy_name = "default"
        price_per_km = 1.5
        approval_threshold_km = None
        # 新增统计
        direct_list: List[str] = []
        supervisor_list: List[str] = []
        finance_list: List[str] = []
        unverified_list: List[str] = []
        unverified_count = 0
        unverified_breakdown: Dict[str, int] = {}
        policy_breakdown: Dict[str, Dict] = {}

        for trip in trips:
            emp = employee_name or trip.employee_name
            dept = department
            if not dept and self.employee_resolver is not None and hasattr(self.employee_resolver, "resolve_department"):
                dept = self.employee_resolver.resolve_department(emp)
            seg_reports, stays = self.build_segment_reports(
                trip, check_route=check_route, employee_name=emp, department=dept,
            )
            all_seg_reports.extend(seg_reports)
            all_stays.extend(stays)
            total_dist += sum(
                (s.moving_distance_km if s.moving_distance_km > 0 else s.total_distance_km)
                for s in trip.segments
            )

            for seg in trip.segments:
                seg_dist = seg.moving_distance_km if seg.moving_distance_km > 0 else seg.total_distance_km
                max_single = max(max_single, seg_dist)
                total_moving += seg.moving_time
                if seg.start_time:
                    first_time = min(first_time, seg.start_time) if first_time else seg.start_time
                if seg.end_time:
                    last_time = max(last_time, seg.end_time) if last_time else seg.end_time

        for seg_idx, sr in enumerate(all_seg_reports, 1):
            seg_label = f"{sr.date} 第{seg_idx}段 ({sr.distance_km:.1f}km)"
            seg_label_short = f"{sr.date} 第{seg_idx}段"

            if sr.compliance_warning:
                violations.append(
                    f"{seg_label} 超出审批标准"
                )
            if sr.reimbursement:
                total_reimburse += sr.reimbursement.amount
                # 审批流分类
                rb = sr.reimbursement
                amount_str = f"{rb.amount:.2f}元 · 政策[{sr.policy_name}]"
                if rb.needs_approval:
                    for reason in rb.approval_reasons:
                        approval_list.append(f"{seg_label}: {reason}")
                    if rb.approval_stage == "supervisor":
                        supervisor_list.append(f"{seg_label} · {amount_str} · {'；'.join(rb.approval_reasons)}")
                    elif rb.approval_stage == "finance":
                        finance_list.append(f"{seg_label} · {amount_str} · {'；'.join(rb.approval_reasons)}")
                else:
                    direct_list.append(f"{seg_label} · {amount_str}")
                # 政策汇总
                pname = sr.policy_name
                if pname not in policy_breakdown:
                    policy_breakdown[pname] = {"count": 0, "distance": 0.0, "amount": 0.0, "price_per_km": rb.price_per_km}
                policy_breakdown[pname]["count"] += 1
                policy_breakdown[pname]["distance"] += sr.distance_km
                policy_breakdown[pname]["amount"] += rb.amount
                # 保留首个非默认政策作为展示
                if pname != "default" and policy_name == "default":
                    policy_name = pname
                    price_per_km = rb.price_per_km
                    approval_threshold_km = rb.approval_threshold_km
                if pname == "default" and policy_name == "default":
                    approval_threshold_km = rb.approval_threshold_km

            # 未核验统计
            if sr.is_unverified:
                unverified_count += 1
                reason = sr.unverified_reason_display
                unverified_breakdown[reason] = unverified_breakdown.get(reason, 0) + 1
                unverified_list.append(f"{seg_label_short} · 原因: {reason}")

        name = employee_name or (trips[0].employee_name if trips else "未知员工")
        date_range = ""
        if first_time and last_time:
            date_range = f"{first_time.strftime('%Y-%m-%d')} ~ {last_time.strftime('%Y-%m-%d')}"

        total_duration = (last_time - first_time) if (first_time and last_time) else timedelta(0)

        # 格式化政策汇总
        for pname in policy_breakdown:
            policy_breakdown[pname]["amount"] = round(policy_breakdown[pname]["amount"], 2)
            policy_breakdown[pname]["distance"] = round(policy_breakdown[pname]["distance"], 2)

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
            total_reimbursement_amount=round(total_reimburse, 2),
            approval_required=approval_list,
            policy_name=policy_name,
            price_per_km=price_per_km,
            approval_threshold_km=approval_threshold_km,
            direct_reimburse=direct_list,
            supervisor_approval=supervisor_list,
            finance_review=finance_list,
            unverified_segments=unverified_list,
            unverified_count=unverified_count,
            unverified_breakdown=unverified_breakdown,
            policy_breakdown=policy_breakdown,
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
            if sr.route_check and sr.route_check.verified:
                rec_dist = f"{sr.route_check.recommended_distance_km:.2f}"
                dev_pct = f"{sr.route_check.deviation_percent:.1f}"
                route_status = sr.route_check.status_label
                unverified_reason = ""
            elif sr.route_check and not sr.route_check.verified:
                rec_dist = ""
                dev_pct = ""
                route_status = "未核验"
                unverified_reason = sr.route_check.unverified_reason_display
            else:
                rec_dist = ""
                dev_pct = ""
                route_status = "未核验"
                unverified_reason = sr.unverified_reason_display

            reimburse_amount = ""
            price_per_km = ""
            is_night = ""
            approval_reasons = ""
            approval_stage = ""
            wl_start = ""
            wl_end = ""
            policy_name = sr.policy_name
            if sr.reimbursement:
                reimburse_amount = f"{sr.reimbursement.amount:.2f}"
                price_per_km = f"{sr.reimbursement.price_per_km:.2f}"
                is_night = "是" if sr.reimbursement.is_night else "否"
                approval_reasons = "; ".join(sr.reimbursement.approval_reasons)
                stage_map = {"direct": "可直接报销", "supervisor": "待主管审批", "finance": "待财务复核"}
                approval_stage = stage_map.get(sr.reimbursement.approval_stage, "")
                wl_start = sr.reimbursement.whitelist_start or ""
                wl_end = sr.reimbursement.whitelist_end or ""

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
                "行驶时长": _format_timedelta(sr.duration),
                "平均速度(km/h)": f"{sr.avg_speed_kmh:.1f}",
                "合规状态": sr.compliance_status,
                "最短路线(km)": rec_dist,
                "偏差(%)": dev_pct,
                "路线状态": route_status,
                "未核验原因": unverified_reason,
                "报销政策": policy_name,
                "审批状态": approval_stage,
                "可报销金额(元)": reimburse_amount,
                "单价(元/km)": price_per_km,
                "夜间出行": is_night,
                "需审批原因": approval_reasons,
                "起点白名单": wl_start,
                "终点白名单": wl_end,
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

        route_map: Dict[Tuple[str, int], RouteComparison] = {}
        trip_id_map: Dict[Tuple[str, int], int] = {}
        if report:
            for idx, sr in enumerate(report.segments):
                if sr.route_check:
                    key = (sr.trip_source_id, sr.segment.segment_id)
                    route_map[key] = sr.route_check
                    trip_id_map[key] = idx

        seg_global_idx = 0
        for trip in trips:
            trip_source = make_trip_source_id(trip.file_path) if trip.file_path else f"trip_{id(trip)}"
            for seg in trip.segments:
                color = colors[seg_global_idx % len(colors)]
                seg_global_idx += 1

                if not seg.points:
                    continue

                MAX_MAP_POINTS = 500
                if len(seg.points) > MAX_MAP_POINTS:
                    step = max(1, len(seg.points) // MAX_MAP_POINTS)
                    sampled = seg.points[::step]
                    if sampled[-1] is not seg.points[-1]:
                        sampled.append(seg.points[-1])
                else:
                    sampled = seg.points
                latlons = [[p.latitude, p.longitude] for p in sampled]
                seg_display_km = seg.moving_distance_km if seg.moving_distance_km > 0 else seg.total_distance_km

                display_seg_num = trip_id_map.get((trip_source, seg.segment_id), seg.segment_id) + 1
                folium.PolyLine(
                    locations=latlons,
                    color=color,
                    weight=4,
                    opacity=0.85,
                    tooltip=f"第{display_seg_num}段 · {seg_display_km:.1f}km · 行驶{_format_timedelta(seg.moving_time)}",
                ).add_to(m)

                sp = seg.points[0]
                folium.Marker(
                    location=[sp.latitude, sp.longitude],
                    icon=folium.Icon(color="green", icon="play"),
                    popup=f"起点<br>时间: {sp.time.strftime('%H:%M:%S')}<br>第{display_seg_num}段",
                ).add_to(m)

                ep = seg.points[-1]
                folium.Marker(
                    location=[ep.latitude, ep.longitude],
                    icon=folium.Icon(color="red", icon="flag"),
                    popup=f"终点<br>时间: {ep.time.strftime('%H:%M:%S')}<br>里程: {seg.total_distance_km:.2f}km",
                ).add_to(m)

                rc = route_map.get((trip_source, seg.segment_id))
                if rc and rc.verified and rc.recommended_route_polyline:
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
                                tooltip=f"推荐路线(第{display_seg_num}段) {rc.recommended_distance_km:.1f}km · 偏差{rc.deviation_percent:.1f}%",
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
{% if report.policy_name %}
**报销政策**: {{ report.policy_name }}
{% endif %}
{% if report.price_per_km is not none and report.price_per_km > 0 %}
**参考单价**: {{ "%.2f"|format(report.price_per_km) }} 元/km
{% endif %}
{% if report.approval_threshold_km is not none %}
**审批阈值**: 单段 > {{ "%.1f"|format(report.approval_threshold_km) }} km
{% endif %}

---

## 📊 行程总览

| 序号 | 日期 | 起点 | 终点 | 里程(km) | 行驶时长 | 均速(km/h) | 推荐路线(km) | 偏差(%) | 路线 | 政策 | 报销(元) | 夜间 | 审批 | 合规 |
|------|------|------|------|----------|----------|------------|--------------|---------|------|------|----------|------|------|------|
{% for s in report.segments %}| {{ loop.index }} | {{ s.date }} | {{ s.start_coord_label }} | {{ s.end_coord_label }} | {{ "%.2f"|format(s.distance_km) }} | {{ s.duration | format_td }} | {{ "%.1f"|format(s.avg_speed_kmh) }} | {{ s.rec_distance_display }} | {{ s.deviation_display }} | {{ s.route_status_display }} | {{ s.policy_name }} | {% if s.reimbursement %}{{ "%.2f"|format(s.reimbursement.amount) }}{% else %}-{% endif %} | {% if s.reimbursement %}{{ '是' if s.reimbursement.is_night else '否' }}{% else %}-{% endif %} | {% if s.reimbursement %}{{ '主管' if s.reimbursement.approval_stage == 'supervisor' else ('财务' if s.reimbursement.approval_stage == 'finance' else '直接') }}{% else %}-{% endif %} | {{ s.compliance_status }} |
{% endfor %}

---

## 📈 统计汇总

- **行程段数**: {{ report.segments|length }} 段
- **停留/办事次数**: {{ report.stay_points|length }} 次
- **总里程**: **{{ "%.2f"|format(report.total_distance_km) }} km**
- **行驶时长**: {{ report.total_moving_time | format_td }}
- **最大单段里程**: {{ "%.2f"|format(report.max_single_segment_km) }} km
- **可报销总额**: **{{ "%.2f"|format(report.total_reimbursement_amount) }} 元**
{% if report.unverified_count > 0 %}
- **未核验段数**: {{ report.unverified_count }} 段
{% for reason, cnt in report.unverified_breakdown.items() %}  - {{ reason }}: {{ cnt }} 段
{% endfor %}
{% endif %}

{% if report.policy_breakdown %}
### 📋 政策使用汇总

| 政策名 | 段数 | 里程(km) | 金额(元) | 单价(元/km) |
|--------|------|----------|----------|-------------|
{% for pname, info in report.policy_breakdown.items() %}| {{ pname }} | {{ info.count }} | {{ "%.2f"|format(info.distance) }} | {{ "%.2f"|format(info.amount) }} | {{ "%.2f"|format(info.price_per_km) }} |
{% endfor %}
{% endif %}

{% if report.direct_reimburse or report.supervisor_approval or report.finance_review %}
---

## 💰 报销审批清单

{% if report.direct_reimburse %}
### ✅ 可直接报销 ({{ report.direct_reimburse|length }} 段)
{% for item in report.direct_reimburse %}
- {{ item }}
{% endfor %}
{% endif %}

{% if report.finance_review %}
### 📋 待财务复核 ({{ report.finance_review|length }} 段)
{% for item in report.finance_review %}
- {{ item }}
{% endfor %}
{% endif %}

{% if report.supervisor_approval %}
### 🔴 待主管审批 ({{ report.supervisor_approval|length }} 段)
{% for item in report.supervisor_approval %}
- {{ item }}
{% endfor %}
{% endif %}
{% endif %}

{% if report.unverified_segments %}
---

## ⚠️ 未核验段清单 ({{ report.unverified_count }} 段)

{% for item in report.unverified_segments %}
- {{ item }}
{% endfor %}
{% endif %}

{% if report.compliance_violations %}
---

## 🚨 里程超限 ({{ report.compliance_violations|length }} 项)
{% for v in report.compliance_violations %}
- {{ v }}
{% endfor %}
{% endif %}

---

## 🔍 各段行程详情

{% for s in report.segments %}
### 第 {{ loop.index }} 段 · {{ s.date }} · 政策[{{ s.policy_name }}]

- **起点**: {{ s.start_coord_label }}{% if s.reimbursement and s.reimbursement.whitelist_start %} ({{ s.reimbursement.whitelist_start }}){% endif %}
- **终点**: {{ s.end_coord_label }}{% if s.reimbursement and s.reimbursement.whitelist_end %} ({{ s.reimbursement.whitelist_end }}){% endif %}
- **起止时间**: {{ s.segment.start_time.strftime('%H:%M:%S') if s.segment.start_time else 'N/A' }} ~ {{ s.segment.end_time.strftime('%H:%M:%S') if s.segment.end_time else 'N/A' }}
- **实际里程**: {{ "%.2f"|format(s.distance_km) }} km
- **行驶时长**: {{ s.duration | format_td }}
- **平均速度**: {{ "%.1f"|format(s.avg_speed_kmh) }} km/h
{% if s.is_unverified %}
- **路线核验**: ⚠️ 未核验 · 原因: {{ s.unverified_reason_display }}
{% else %}
{% if s.route_check and s.route_check.verified %}
- **推荐最短路线**: {{ "%.2f"|format(s.route_check.recommended_distance_km) }} km
- **路线偏差**: {{ "%.1f"|format(s.route_check.deviation_percent) }} % {{ '⚠️ 疑似绕路' if s.route_check.is_detour else '✓ 正常' }}
{% endif %}
{% endif %}
- **合规状态**: {{ s.compliance_status }}
{% if s.reimbursement %}
- **报销政策**: {{ s.reimbursement.policy_name }}
- **可报销金额**: {{ "%.2f"|format(s.reimbursement.amount) }} 元 (单价 {{ "%.2f"|format(s.reimbursement.price_per_km) }} 元/km{% if s.reimbursement.is_night %}, 夜间×{{ "%.1f"|format(s.reimbursement.night_multiplier) }}{% endif %})
- **审批状态**: {{ '待主管审批' if s.reimbursement.approval_stage == 'supervisor' else ('待财务复核' if s.reimbursement.approval_stage == 'finance' else '可直接报销') }}
{% if s.reimbursement.approval_reasons %}
- **需审批原因**: {{ s.reimbursement.approval_reasons | join('；') }}
{% endif %}
{% endif %}

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
    max-width: 1300px;
    margin: 0 auto;
    padding: 28px 32px;
    color: #2c3e50;
    line-height: 1.6;
    background: #f7f9fc;
  }
  h1 { color: #2980b9; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #27ae60; margin-top: 28px; border-left: 5px solid #27ae60; padding-left: 10px; }
  h3 { color: #8e44ad; }
  h4 { color: #2980b9; margin-top: 14px; }
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
    padding: 10px 10px;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
    font-size: 13px;
  }
  th { background: linear-gradient(to bottom, #3498db, #2980b9); color: #fff; font-weight: 600; white-space: nowrap; }
  tr:hover td { background: #f8fbff; }
  tr:last-child td { border-bottom: none; }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin: 14px 0;
  }
  .stat-card {
    background: #fff;
    padding: 14px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    border-top: 3px solid #3498db;
  }
  .stat-card .label { font-size: 13px; color: #7f8c8d; }
  .stat-card .value { font-size: 22px; font-weight: 700; color: #2c3e50; margin-top: 4px; }
  .warn { background: #fff8e1 !important; color: #e67e22; font-weight: 600; }
  .danger { background: #ffebee !important; color: #c0392b; font-weight: 700; }
  .ok { color: #27ae60; font-weight: 600; }
  .muted { color: #95a5a6; }
  .policy-tag { display:inline-block; padding:2px 8px; background:#eef2ff; color:#4f46e5; border-radius:10px; font-size:12px; font-weight:600; }
  .tag-direct { background:#dcfce7; color:#166534; }
  .tag-finance { background:#fef9c3; color:#854d0e; }
  .tag-supervisor { background:#fee2e2; color:#991b1b; }
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
  .alert-warning { background: #fff3cd; border-color: #ffeaa7; color: #856404; }
  .alert-success { background: #d4edda; border-color: #c3e6cb; color: #155724; }
  .alert-info { background: #d1ecf1; border-color: #bee5eb; color: #0c5460; }
  .list-section { margin: 8px 0; }
  .list-section ul { margin: 8px 0 0 20px; padding: 0; }
  .list-section li { margin: 4px 0; font-size: 14px; }
  .footer { margin-top: 40px; text-align: center; color: #95a5a6; font-size: 12px; padding: 18px; border-top: 1px solid #ecf0f1; }
</style>
</head>
<body>

<h1>📋 {{ report.report_title }}</h1>

<div class="header-info">
  <div><b>员工姓名:</b> {{ report.employee_name }}</div>
  {% if report.date_range %}<div><b>日期范围:</b> {{ report.date_range }}</div>{% endif %}
  <div><b>生成时间:</b> {{ report.generated_at }}</div>
  {% if report.policy_name %}<div><b>报销政策:</b> {{ report.policy_name }}</div>{% endif %}
  {% if report.price_per_km is not none %}<div><b>参考单价:</b> {{ "%.2f"|format(report.price_per_km) }} 元/km</div>{% endif %}
  {% if report.approval_threshold_km is not none %}<div><b>审批阈值:</b> 单段 > {{ "%.1f"|format(report.approval_threshold_km) }} km</div>{% endif %}
</div>

<h2>📊 行程总览表</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>日期</th><th>起点</th><th>终点</th><th>里程(km)</th>
      <th>行驶时长</th><th>均速</th><th>推荐</th><th>偏差</th><th>路线</th>
      <th>政策</th><th>报销(元)</th><th>夜间</th><th>审批</th><th>合规</th>
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
      <td>{{ s.rec_distance_display }}</td>
      <td class="{% if s.route_check and s.route_check.verified and s.route_check.deviation_percent > DEVIATION_THRESHOLD %}warn{% endif %}">{{ s.deviation_display }}</td>
      <td class="{% if s.route_check and s.route_check.verified and s.route_check.is_detour %}warn{% else %}ok{% endif %}">{{ s.route_status_display }}</td>
      <td><span class="policy-tag">{{ s.policy_name }}</span></td>
      <td><b>{% if s.reimbursement %}{{ "%.2f"|format(s.reimbursement.amount) }}{% else %}-{% endif %}</b></td>
      <td>{% if s.reimbursement and s.reimbursement.is_night %}<span class="warn">是</span>{% else %}否{% endif %}</td>
      <td>{% if s.reimbursement %}<span class="policy-tag {% if s.reimbursement.approval_stage == 'supervisor' %}tag-supervisor{% elif s.reimbursement.approval_stage == 'finance' %}tag-finance{% else %}tag-direct{% endif %}">
        {{ '主管' if s.reimbursement.approval_stage == 'supervisor' else ('财务' if s.reimbursement.approval_stage == 'finance' else '直接') }}
      </span>{% else %}-{% endif %}</td>
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
  <div class="stat-card"><div class="label">行驶时长</div><div class="value">{{ report.total_moving_time | format_td }}</div></div>
  <div class="stat-card"><div class="label">最大单段里程</div><div class="value">{{ "%.2f"|format(report.max_single_segment_km) }} km</div></div>
  {% if report.total_reimbursement_amount is not none %}
  <div class="stat-card" style="border-top-color:#27ae60;"><div class="label">可报销总额</div><div class="value" style="color:#27ae60;">{{ "%.2f"|format(report.total_reimbursement_amount) }} 元</div></div>
  {% endif %}
  {% if report.unverified_count > 0 %}
  <div class="stat-card" style="border-top-color:#f39c12;"><div class="label">未核验段</div><div class="value" style="color:#e67e22;">{{ report.unverified_count }} 段</div></div>
  {% endif %}
</div>

{% if report.unverified_count > 0 %}
<div class="alert alert-warning">
  <b>⚠️ 未核验段统计:</b>
  <ul style="margin:6px 0 0 20px;">
  {% for reason, cnt in report.unverified_breakdown.items() %}
    <li>{{ reason }}: {{ cnt }} 段</li>
  {% endfor %}
  </ul>
</div>
{% endif %}

{% if report.policy_breakdown %}
<h3>📋 政策使用汇总</h3>
<table>
  <thead>
    <tr><th>政策名</th><th>段数</th><th>里程(km)</th><th>金额(元)</th><th>单价(元/km)</th></tr>
  </thead>
  <tbody>
  {% for pname, info in report.policy_breakdown.items() %}
    <tr>
      <td><span class="policy-tag">{{ pname }}</span></td>
      <td>{{ info.count }}</td>
      <td>{{ "%.2f"|format(info.distance) }}</td>
      <td><b>{{ "%.2f"|format(info.amount) }}</b></td>
      <td>{{ "%.2f"|format(info.price_per_km) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if report.direct_reimburse or report.supervisor_approval or report.finance_review %}
<h2>💰 报销审批清单</h2>

{% if report.direct_reimburse %}
<div class="alert alert-success list-section">
  <b>✅ 可直接报销 ({{ report.direct_reimburse|length }} 段):</b>
  <ul>
  {% for item in report.direct_reimburse %}<li>{{ item }}</li>{% endfor %}
  </ul>
</div>
{% endif %}

{% if report.finance_review %}
<div class="alert alert-warning list-section">
  <b>📋 待财务复核 ({{ report.finance_review|length }} 段):</b>
  <ul>
  {% for item in report.finance_review %}<li>{{ item }}</li>{% endfor %}
  </ul>
</div>
{% endif %}

{% if report.supervisor_approval %}
<div class="alert alert-danger list-section">
  <b>🔴 待主管审批 ({{ report.supervisor_approval|length }} 段):</b>
  <ul>
  {% for item in report.supervisor_approval %}<li>{{ item }}</li>{% endfor %}
  </ul>
</div>
{% endif %}
{% endif %}

{% if report.unverified_segments %}
<div class="alert alert-info list-section">
  <b>⚠️ 未核验段清单 ({{ report.unverified_count }} 段):</b>
  <ul>
  {% for item in report.unverified_segments %}<li>{{ item }}</li>{% endfor %}
  </ul>
</div>
{% endif %}

{% if report.compliance_violations %}
<div class="alert alert-danger list-section">
  <b>🚨 里程超限 ({{ report.compliance_violations|length }} 项):</b>
  <ul>
  {% for v in report.compliance_violations %}<li>{{ v }}</li>{% endfor %}
  </ul>
</div>
{% endif %}

<h2>🔍 各段行程详情</h2>

{% for s in report.segments %}
<div class="seg-box">
  <h3>🚗 第 {{ loop.index }} 段 · {{ s.date }} · <span class="policy-tag">{{ s.policy_name }}</span></h3>
  <div class="seg-meta">
    <div><span>起点坐标</span><b>{{ s.start_coord_label }}</b>{% if s.reimbursement and s.reimbursement.whitelist_start %} <span class="ok" style="font-size:12px;">({{ s.reimbursement.whitelist_start }})</span>{% endif %}</div>
    <div><span>终点坐标</span><b>{{ s.end_coord_label }}</b>{% if s.reimbursement and s.reimbursement.whitelist_end %} <span class="ok" style="font-size:12px;">({{ s.reimbursement.whitelist_end }})</span>{% endif %}</div>
    <div><span>起止时间</span><b>{{ s.segment.start_time.strftime('%H:%M:%S') if s.segment.start_time else 'N/A' }} ~ {{ s.segment.end_time.strftime('%H:%M:%S') if s.segment.end_time else 'N/A' }}</b></div>
    <div><span>实际里程</span><b>{{ "%.2f"|format(s.distance_km) }} km</b></div>
    <div><span>行驶时长</span><b>{{ s.duration | format_td }}</b></div>
    <div><span>平均速度</span><b>{{ "%.1f"|format(s.avg_speed_kmh) }} km/h</b></div>
    {% if s.is_unverified %}
    <div class="muted"><span>推荐最短路线</span><b>-</b></div>
    <div class="alert" style="padding:6px 10px;"><span>路线核验</span><b>⚠️ 未核验 · 原因: {{ s.unverified_reason_display }}</b></div>
    {% else %}
    {% if s.route_check and s.route_check.verified %}
    <div><span>推荐最短路线</span><b>{{ "%.2f"|format(s.route_check.recommended_distance_km) }} km</b></div>
    <div class="{% if s.route_check.is_detour %}alert{% endif %}" style="{% if s.route_check.is_detour %}padding:6px 10px;{% endif %}"><span>路线偏差</span><b>{{ "%.1f"|format(s.route_check.deviation_percent) }} % {{ '⚠️ 疑似绕路' if s.route_check.is_detour else '✓ 正常' }}</b></div>
    {% endif %}
    {% endif %}
    <div class="{% if s.compliance_warning %}alert-danger alert{% endif %}" style="{% if s.compliance_warning %}padding:6px 10px;{% endif %}"><span>合规状态</span><b>{{ s.compliance_status }}</b></div>
    {% if s.reimbursement %}
    <div style="background:#e8f5e9;"><span>可报销金额</span><b style="color:#2e7d32;">{{ "%.2f"|format(s.reimbursement.amount) }} 元</b></div>
    <div><span>报销政策</span><b>{{ s.reimbursement.policy_name }}</b></div>
    <div><span>报销单价</span><b>{{ "%.2f"|format(s.reimbursement.price_per_km) }} 元/km{% if s.reimbursement.is_night %} (夜间×{{ "%.1f"|format(s.reimbursement.night_multiplier) }}){% endif %}</b></div>
    <div><span>审批状态</span><b>
      <span class="policy-tag {% if s.reimbursement.approval_stage == 'supervisor' %}tag-supervisor{% elif s.reimbursement.approval_stage == 'finance' %}tag-finance{% else %}tag-direct{% endif %}">
        {{ '待主管审批' if s.reimbursement.approval_stage == 'supervisor' else ('待财务复核' if s.reimbursement.approval_stage == 'finance' else '可直接报销') }}
      </span>
    </b></div>
    {% if s.reimbursement.approval_reasons %}
    <div class="alert" style="padding:6px 10px;"><span>需审批原因</span><b>{{ s.reimbursement.approval_reasons|join('；') }}</b></div>
    {% endif %}
    {% endif %}
  </div>

  {% if s.segment.stay_points %}
  <h4>⏸️ 停留/办事点</h4>
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
  本报告由 TripLog 行程管理工具自动生成 · 数据仅供报销参考 · GPS轨迹数据可能存在误差
</div>
</body>
</html>
"""
