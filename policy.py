from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from geopy.distance import geodesic

from trip_parser import TripSegment


@dataclass
class WhitelistLocation:
    name: str
    latitude: float
    longitude: float
    radius_m: float = 200.0
    category: str = "office"

    def matches(self, lat: float, lon: float) -> bool:
        dist = geodesic((self.latitude, self.longitude), (lat, lon)).meters
        return dist <= self.radius_m


@dataclass
class NightRule:
    start: time = time(22, 0)
    end: time = time(6, 0)
    multiplier: float = 1.5
    allowance_per_trip: float = 0.0


@dataclass
class ReimbursementPolicy:
    name: str = "default"
    price_per_km: float = 1.5
    approval_threshold_km: float = 300.0
    night_rule: NightRule = field(default_factory=NightRule)
    whitelist: List[WhitelistLocation] = field(default_factory=list)
    applies_to_employees: List[str] = field(default_factory=list)
    applies_to_departments: List[str] = field(default_factory=list)
    applies_to_cities: List[str] = field(default_factory=list)
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    def applies_to(self, employee: str, date_str: str, city: str = "", department: str = "") -> bool:
        if self.applies_to_employees and employee not in self.applies_to_employees:
            return False
        if self.applies_to_departments and department not in self.applies_to_departments:
            return False
        if self.applies_to_cities:
            if not city:
                return False
            if city not in self.applies_to_cities:
                return False
        if self.date_start and date_str < self.date_start:
            return False
        if self.date_end and date_str > self.date_end:
            return False
        return True


@dataclass
class SegmentReimbursement:
    amount: float = 0.0
    price_per_km: float = 0.0
    is_night: bool = False
    night_multiplier: float = 1.0
    needs_approval: bool = False
    approval_reasons: List[str] = field(default_factory=list)
    whitelist_start: Optional[str] = None
    whitelist_end: Optional[str] = None
    policy_name: str = "default"
    approval_stage: str = "direct"  # direct(直接报销) / supervisor(待主管) / finance(待财务)
    approval_threshold_km: float = 300.0


class PolicyEngine:
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        self.verbose = verbose
        self.policies: List[ReimbursementPolicy] = [ReimbursementPolicy()]
        if config_path and Path(config_path).exists():
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        p = Path(config_path)
        if p.suffix.lower() in (".csv",):
            self._load_csv(p)
        elif p.suffix.lower() in (".json",):
            self._load_json(p)
        else:
            try:
                self._load_csv(p)
            except Exception:
                pass

    def _load_csv(self, p: Path) -> None:
        policies: Dict[str, ReimbursementPolicy] = {}
        whitelist_rows: List[dict] = []
        with open(p, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {
                    (k.strip() if k else ""): (v.strip() if v else "")
                    for k, v in row.items()
                    if k is not None
                }
                pname = row.get("政策名称") or row.get("policy") or "default"
                if pname not in policies:
                    pol = ReimbursementPolicy(name=pname)
                    if row.get("每公里单价(元)") or row.get("price_per_km"):
                        pol.price_per_km = float(
                            row.get("每公里单价(元)") or row.get("price_per_km") or 1.5
                        )
                    if row.get("审批阈值(km)") or row.get("approval_threshold_km"):
                        pol.approval_threshold_km = float(
                            row.get("审批阈值(km)") or row.get("approval_threshold_km") or 300
                        )
                    if row.get("适用员工") or row.get("employees"):
                        pol.applies_to_employees = [
                            x.strip() for x in
                            (row.get("适用员工") or row.get("employees") or "").split(";")
                            if x.strip()
                        ]
                    if row.get("适用部门") or row.get("departments") or row.get("department"):
                        pol.applies_to_departments = [
                            x.strip() for x in
                            (row.get("适用部门") or row.get("departments") or row.get("department") or "").split(";")
                            if x.strip()
                        ]
                    if row.get("适用城市") or row.get("cities"):
                        pol.applies_to_cities = [
                            x.strip() for x in
                            (row.get("适用城市") or row.get("cities") or "").split(";")
                            if x.strip()
                        ]
                    if row.get("生效日期") or row.get("date_start"):
                        pol.date_start = row.get("生效日期") or row.get("date_start")
                    if row.get("截止日期") or row.get("date_end"):
                        pol.date_end = row.get("截止日期") or row.get("date_end")
                    if row.get("夜间开始") or row.get("night_start"):
                        t = row.get("夜间开始") or row.get("night_start") or "22:00"
                        pol.night_rule.start = time.fromisoformat(t)
                    if row.get("夜间结束") or row.get("night_end"):
                        t = row.get("夜间结束") or row.get("night_end") or "06:00"
                        pol.night_rule.end = time.fromisoformat(t)
                    if row.get("夜间加价倍数") or row.get("night_multiplier"):
                        pol.night_rule.multiplier = float(
                            row.get("夜间加价倍数") or row.get("night_multiplier") or 1.5
                        )
                    policies[pname] = pol

                if row.get("白名单地点") or row.get("whitelist_name"):
                    whitelist_rows.append(row)

        for row in whitelist_rows:
            pname = row.get("政策名称") or row.get("policy") or "default"
            pol = policies.get(pname)
            if not pol:
                continue
            name = row.get("白名单地点") or row.get("whitelist_name") or ""
            lat_str = row.get("纬度") or row.get("latitude") or row.get("lat")
            lon_str = row.get("经度") or row.get("longitude") or row.get("lon")
            if not name or not lat_str or not lon_str:
                continue
            radius = float(row.get("半径(m)") or row.get("radius_m") or 200)
            cat = row.get("类别") or row.get("category") or "office"
            try:
                pol.whitelist.append(WhitelistLocation(
                    name=name,
                    latitude=float(lat_str),
                    longitude=float(lon_str),
                    radius_m=radius,
                    category=cat,
                ))
            except ValueError:
                pass

        self.policies = list(policies.values()) if policies else [ReimbursementPolicy()]
        if self.verbose:
            from rich.console import Console
            console = Console()
            console.print(f"[dim]已加载报销政策: {len(self.policies)} 条[/dim]")

    def _load_json(self, p: Path) -> None:
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        policies: List[ReimbursementPolicy] = []
        for entry in data.get("policies", []):
            pol = ReimbursementPolicy(
                name=entry.get("name", "default"),
                price_per_km=entry.get("price_per_km", 1.5),
                approval_threshold_km=entry.get("approval_threshold_km", 300.0),
            )
            if "night_rule" in entry:
                nr = entry["night_rule"]
                pol.night_rule = NightRule(
                    start=time.fromisoformat(nr.get("start", "22:00")),
                    end=time.fromisoformat(nr.get("end", "06:00")),
                    multiplier=nr.get("multiplier", 1.5),
                    allowance_per_trip=nr.get("allowance_per_trip", 0.0),
                )
            for wl in entry.get("whitelist", []):
                pol.whitelist.append(WhitelistLocation(
                    name=wl.get("name", ""),
                    latitude=wl.get("latitude", 0),
                    longitude=wl.get("longitude", 0),
                    radius_m=wl.get("radius_m", 200.0),
                    category=wl.get("category", "office"),
                ))
            pol.applies_to_employees = entry.get("applies_to_employees", [])
            pol.applies_to_departments = entry.get("applies_to_departments", [])
            pol.applies_to_cities = entry.get("applies_to_cities", [])
            pol.date_start = entry.get("date_start")
            pol.date_end = entry.get("date_end")
            policies.append(pol)
        if policies:
            self.policies = policies

    def find_policy(self, employee: str, date_str: str, city: str = "", department: str = "") -> ReimbursementPolicy:
        best = None
        best_score = -1
        for pol in self.policies:
            if not pol.applies_to(employee, date_str, city, department):
                continue
            score = 0
            if pol.applies_to_employees:
                score += 10
            if pol.applies_to_departments:
                score += 8
            if pol.applies_to_cities:
                score += 5
            if pol.date_start or pol.date_end:
                score += 3
            if score > best_score:
                best_score = score
                best = pol
        return best or self.policies[0]

    def compute_reimbursement(
        self,
        segment: TripSegment,
        employee: str,
        date_str: str = "",
        city: str = "",
        department: str = "",
    ) -> SegmentReimbursement:
        if not date_str and segment.start_time:
            date_str = segment.start_time.strftime("%Y-%m-%d")

        policy = self.find_policy(employee, date_str, city, department)
        result = SegmentReimbursement(
            price_per_km=policy.price_per_km,
            policy_name=policy.name,
            approval_threshold_km=policy.approval_threshold_km,
        )

        dist_km = segment.moving_distance_km if segment.moving_distance_km > 0 else segment.total_distance_km

        is_night = False
        if segment.start_time and segment.end_time:
            st = segment.start_time.time()
            et = segment.end_time.time()
            ns = policy.night_rule.start
            ne = policy.night_rule.end
            def _in_night(t: time) -> bool:
                if ns <= ne:
                    return ns <= t <= ne
                return t >= ns or t <= ne
            is_night = _in_night(st) or _in_night(et)

        if is_night:
            result.is_night = True
            result.night_multiplier = policy.night_rule.multiplier
            effective_price = policy.price_per_km * policy.night_rule.multiplier
        else:
            effective_price = policy.price_per_km

        result.amount = round(dist_km * effective_price, 2)

        threshold = policy.approval_threshold_km
        if dist_km > threshold:
            result.needs_approval = True
            result.approval_stage = "supervisor"
            result.approval_reasons.append(f"单段里程{dist_km:.1f}km超过{threshold:.0f}km审批阈值")
        elif dist_km > threshold * 0.5:
            result.needs_approval = True
            result.approval_stage = "finance"
            result.approval_reasons.append(f"单段里程{dist_km:.1f}km达到{threshold:.0f}km阈值的50%")

        if is_night and policy.night_rule.allowance_per_trip > 0:
            result.amount = round(result.amount + policy.night_rule.allowance_per_trip, 2)
            if not result.needs_approval:
                result.needs_approval = True
                result.approval_stage = "finance"
            result.approval_reasons.append("夜间出行补贴")

        for wl in policy.whitelist:
            if segment.start_point and wl.matches(*segment.start_point):
                result.whitelist_start = wl.name
            if segment.end_point and wl.matches(*segment.end_point):
                result.whitelist_end = wl.name

        if result.whitelist_start and result.whitelist_end:
            if result.needs_approval and len(result.approval_reasons) == 1 and "50%" in result.approval_reasons[0]:
                result.needs_approval = False
                result.approval_stage = "direct"
                result.approval_reasons = []

        return result
