from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel

from trip_parser import EmployeeResolver, TripParser, TripResult
from reporter import Reporter, FullReport

console = Console()

DEFAULT_OSRM = "https://router.project-osrm.org"


def _month_filter(results: List[TripResult], month: Optional[str]) -> List[TripResult]:
    if not month:
        return results
    try:
        year, mon = month.split("-")
        target_ym = (int(year), int(mon))
    except Exception:
        raise click.BadParameter("月份格式错误，请使用 YYYY-MM 格式，例如 2026-06")

    filtered: List[TripResult] = []
    for r in results:
        keep_segs = [
            s for s in r.segments
            if s.start_time and (s.start_time.year, s.start_time.month) == target_ym
        ]
        if not keep_segs:
            continue
        new_r = TripResult(
            file_path=r.file_path,
            employee_name=r.employee_name,
            segments=keep_segs,
        )
        new_r.recompute_totals()
        new_r.moving_time = sum((s.moving_time for s in keep_segs), start=timedelta(0))
        filtered.append(new_r)
    return filtered


def _group_by_employee_and_month(
    results: List[TripResult],
) -> Dict[Tuple[str, str], List[TripResult]]:
    groups: Dict[Tuple[str, str], List[TripResult]] = defaultdict(list)
    seg_buckets: Dict[Tuple[str, str], List] = defaultdict(list)

    for r in results:
        if not r.segments:
            continue
        emp = r.employee_name or "未知员工"
        for seg in r.segments:
            if seg.start_time:
                ym = f"{seg.start_time.year}-{seg.start_time.month:02d}"
            else:
                ym = datetime.now().strftime("%Y-%m")
            seg_buckets[(emp, ym)].append((r, seg))

    for (emp, ym), items in seg_buckets.items():
        by_file: Dict[str, TripResult] = {}
        for orig_result, seg in items:
            key = orig_result.file_path + "|" + ym
            if key not in by_file:
                tr = TripResult(
                    file_path=orig_result.file_path,
                    employee_name=emp,
                )
                by_file[key] = tr
            by_file[key].segments.append(seg)
        for tr in by_file.values():
            for idx, s in enumerate(tr.segments):
                s.segment_id = idx
            tr.recompute_totals()
            tr.moving_time = sum((s.moving_time for s in tr.segments), start=timedelta(0))
            groups[(emp, ym)].append(tr)

    return groups


def _make_resolver(
    employee_name: str,
    mapping_csv: Optional[str],
    verbose: bool,
) -> Optional[EmployeeResolver]:
    if employee_name:
        return None
    if mapping_csv or not employee_name:
        return EmployeeResolver(
            mapping_csv=mapping_csv, default_name=employee_name or "", verbose=verbose
        )
    return None


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option("0.1.0", prog_name="triplog")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """GPS轨迹解析与员工出差行程报告生成工具 (TripLog)

    支持GPX轨迹解析、行程段切分、合规检查、路线对比、批量处理、报告导出等功能。
    """
    if ctx.invoked_subcommand is None:
        banner = Panel.fit(
            "[bold cyan]TripLog[/bold cyan] - GPS轨迹解析 & 员工出差行程报告工具\n\n"
            "[dim]使用 [cyan]triplog --help[/cyan] 查看所有命令\n"
            "常用命令:[/dim]\n"
            "  [green]triplog parse[/green]    解析单个GPX文件并生成报告\n"
            "  [green]triplog report[/green]   根据解析结果生成报告\n"
            "  [green]triplog map[/green]      生成交互式轨迹地图\n"
            "  [green]triplog batch[/green]    批量处理目录下所有GPX文件\n"
            "  [green]triplog export[/green]   导出数据为CSV",
            title="TripLog v0.1.0",
            border_style="cyan",
        )
        console.print(banner)


_EMPLOYEE_OPT = click.option(
    "--employee-name", "--employee", "-e", "employee_name",
    default="", help="员工姓名 (--employee-name 或 --employee 均可)",
)

_MAPPING_OPT = click.option(
    "--employee-mapping", "--mapping", "mapping_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None, help="员工映射CSV (列: 员工姓名,文件名,别名...)，自动识别员工",
)

_CACHE_OPT = click.option(
    "--cache-dir", type=click.Path(file_okay=False, path_type=Path),
    default=None, help="OSRM路线缓存目录（默认内存缓存）",
)

_POLICY_OPT = click.option(
    "--policy", "--policy-config", "policy_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None, help="报销政策配置CSV/JSON (单价、审批阈值、夜间规则、白名单)",
)


@cli.command("parse")
@click.argument("gpx_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_EMPLOYEE_OPT
@_MAPPING_OPT
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="输出报告文件 (.md/.html/.pdf)")
@click.option("--format", "-f", "fmt", type=click.Choice(["md", "markdown", "html", "pdf", "console"]), default="console", help="输出格式 (默认: console)")
@click.option("--check-route/--no-check-route", default=True, help="是否调用OSRM对比最短路线 (默认开启)")
@click.option("--osrm", default=DEFAULT_OSRM, help="自定义OSRM服务地址")
@_CACHE_OPT
@_POLICY_OPT
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
@click.option("--csv", type=click.Path(dir_okay=False, path_type=Path), help="同时导出CSV (财务用)")
def parse_cmd(
    gpx_file: Path,
    employee_name: str,
    mapping_csv: Optional[Path],
    output: Optional[Path],
    fmt: str,
    check_route: bool,
    osrm: str,
    cache_dir: Optional[Path],
    policy_config: Optional[Path],
    verbose: bool,
    csv: Optional[Path],
) -> None:
    """解析单个GPX文件，提取行程段并生成报告。

    GPX_FILE: GPX轨迹文件路径 (支持标准GPX 1.1格式)

    示例:
      triplog parse trip.gpx --employee "张三" -o report.md
      triplog parse trip.gpx -f html -o report.html --no-check-route
      triplog parse trip.gpx --policy policy.csv --csv expense.csv
    """
    try:
        resolver = _make_resolver(employee_name, str(mapping_csv) if mapping_csv else None, verbose)
        emp = employee_name
        if not emp and resolver:
            emp = resolver.resolve(str(gpx_file))

        parser = TripParser(verbose=verbose)
        result = parser.parse(str(gpx_file), employee_name=emp)

        reporter = Reporter(
            verbose=verbose, osrm_base=osrm,
            cache_dir=str(cache_dir) if cache_dir else None,
            policy_config=str(policy_config) if policy_config else None,
        )
        report = reporter.build_full_report([result], employee_name=emp, check_route=check_route)

        _dispatch_output(reporter, report, [result], output, fmt, csv)

    except Exception as e:
        console.print(f"[bold red]❌ 解析失败:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("report")
@click.argument("gpx_files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dir", "-d", "gpx_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="指定目录批量解析GPX文件")
@_EMPLOYEE_OPT
@_MAPPING_OPT
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="输出报告文件 (.md/.html/.pdf)")
@click.option("--format", "-f", "fmt", type=click.Choice(["md", "markdown", "html", "pdf", "console"]), default="console", help="输出格式 (默认: console)")
@click.option("--month", default="", help="仅处理指定月份 (格式: YYYY-MM，如 2026-06)")
@click.option("--split-by-employee/--no-split-by-employee", default=False, help="按员工分别生成报告（默认合并成一份）")
@click.option("--check-route/--no-check-route", default=True, help="是否调用OSRM对比最短路线")
@click.option("--osrm", default=DEFAULT_OSRM, help="自定义OSRM服务地址")
@_CACHE_OPT
@_POLICY_OPT
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
@click.option("--csv", type=click.Path(dir_okay=False, path_type=Path), help="导出报销CSV")
@click.option("--points-csv", type=click.Path(dir_okay=False, path_type=Path), help="导出全部轨迹点CSV")
def report_cmd(
    gpx_files: tuple,
    gpx_dir: Optional[Path],
    employee_name: str,
    mapping_csv: Optional[Path],
    output: Optional[Path],
    fmt: str,
    month: str,
    split_by_employee: bool,
    check_route: bool,
    osrm: str,
    cache_dir: Optional[Path],
    policy_config: Optional[Path],
    verbose: bool,
    csv: Optional[Path],
    points_csv: Optional[Path],
) -> None:
    """根据一个或多个GPX文件生成完整的行程报告。

    示例:
      triplog report trip1.gpx trip2.gpx --employee "李四" -f md -o monthly.md --month 2026-06
      triplog report -d ./gpx_files/ --split-by-employee --mapping employees.csv --policy policy.csv
    """
    try:
        resolver = _make_resolver(employee_name, str(mapping_csv) if mapping_csv else None, verbose)
        parser = TripParser(verbose=verbose)
        results: List[TripResult] = []

        if gpx_dir:
            if verbose:
                console.print(f"[dim]扫描目录: {gpx_dir}[/dim]")
            results.extend(parser.parse_directory(str(gpx_dir), employee_name=employee_name,
                                                  employee_resolver=resolver))

        for gf in gpx_files:
            emp = employee_name
            if not emp and resolver:
                emp = resolver.resolve(str(gf))
            results.append(parser.parse(str(gf), employee_name=emp))

        if not results:
            console.print("[yellow]未找到任何有效的GPX数据[/yellow]")
            sys.exit(0)

        if month:
            results = _month_filter(results, month)
            if not results:
                console.print(f"[yellow]月份 {month} 内无任何行程数据[/yellow]")
                sys.exit(0)

        reporter = Reporter(
            verbose=verbose, osrm_base=osrm,
            cache_dir=str(cache_dir) if cache_dir else None,
            policy_config=str(policy_config) if policy_config else None,
        )

        if split_by_employee and not employee_name:
            groups = _group_by_employee_and_month(results)
            if verbose:
                console.print(f"[dim]按员工+月份分组: {len(groups)} 组[/dim]")
            for (emp, ym), grp in sorted(groups.items()):
                title = f"{emp} {ym} 月度出差行程报告"
                safe_emp = emp.replace(" ", "_").replace("/", "_")
                prefix = f"{safe_emp}_{ym}"
                rpt = reporter.build_full_report(grp, employee_name=emp, report_title=title, check_route=check_route)
                if output:
                    base = Path(output)
                    ext = base.suffix or (".md" if fmt in ("md", "markdown") else ".html" if fmt == "html" else ".pdf" if fmt == "pdf" else "")
                    per_out = base.with_name(f"{prefix}_{base.stem}{ext}")
                else:
                    per_out = None
                _dispatch_output(reporter, rpt, grp, per_out, fmt, None, None)
                if csv:
                    p = Path(csv)
                    reporter.export_csv(rpt, str(p.with_name(f"{prefix}_{p.name}")))
            return

        title = "员工出差月度行程报告" if month else "员工出差行程报告"
        report = reporter.build_full_report(results, employee_name=employee_name, report_title=title, check_route=check_route)

        _dispatch_output(reporter, report, results, output, fmt, csv, points_csv)

    except Exception as e:
        console.print(f"[bold red]❌ 生成报告失败:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("map")
@click.argument("gpx_files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dir", "-d", "gpx_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="指定目录批量解析")
@_EMPLOYEE_OPT
@_MAPPING_OPT
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=Path("trip_map.html"), help="输出HTML地图路径 (默认: trip_map.html)")
@click.option("--with-route/--no-route", default=True, help="是否叠加OSRM推荐路线")
@click.option("--split-by-employee/--no-split-by-employee", default=False, help="按员工分别生成地图")
@click.option("--osrm", default=DEFAULT_OSRM, help="自定义OSRM服务地址")
@_CACHE_OPT
@_POLICY_OPT
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
def map_cmd(
    gpx_files: tuple,
    gpx_dir: Optional[Path],
    employee_name: str,
    mapping_csv: Optional[Path],
    output: Path,
    with_route: bool,
    split_by_employee: bool,
    osrm: str,
    cache_dir: Optional[Path],
    policy_config: Optional[Path],
    verbose: bool,
) -> None:
    """生成交互式轨迹可视化地图 (HTML格式，基于folium/OpenStreetMap)。

    地图特性:
      · 每段行程不同颜色标识
      · 🟢起点 / 🔴终点 / 🔵停留点标记
      · 推荐路线虚线叠加 (需开启 --with-route)
      · 支持鼠标拖拽和缩放

    示例:
      triplog map trip.gpx -o map.html
      triplog map -d ./gpx/ --employee "赵六" --no-route --policy policy.csv
    """
    try:
        resolver = _make_resolver(employee_name, str(mapping_csv) if mapping_csv else None, verbose)
        parser = TripParser(verbose=verbose)
        results: List[TripResult] = []

        if gpx_dir:
            results.extend(parser.parse_directory(str(gpx_dir), employee_name=employee_name,
                                                  employee_resolver=resolver))
        for gf in gpx_files:
            emp = employee_name
            if not emp and resolver:
                emp = resolver.resolve(str(gf))
            results.append(parser.parse(str(gf), employee_name=emp))

        if not results:
            console.print("[yellow]未找到任何有效的GPX数据[/yellow]")
            sys.exit(0)

        reporter = Reporter(
            verbose=verbose, osrm_base=osrm,
            cache_dir=str(cache_dir) if cache_dir else None,
            policy_config=str(policy_config) if policy_config else None,
        )

        if split_by_employee and not employee_name:
            groups = _group_by_employee_and_month(results)
            for (emp, ym), grp in sorted(groups.items()):
                report = None
                if with_route:
                    report = reporter.build_full_report(grp, employee_name=emp, check_route=True)
                safe_emp = emp.replace(" ", "_").replace("/", "_")
                base = Path(output)
                per_out = base.with_name(f"{safe_emp}_{ym}_{base.name}")
                out_path = reporter.generate_map(grp, str(per_out), report=report)
                console.print(f"[green]✅ {emp} 地图:[/green] {out_path}")
            return

        report = None
        if with_route:
            report = reporter.build_full_report(results, employee_name=employee_name, check_route=True)
        out_path = reporter.generate_map(results, str(output), report=report)

        console.print(f"[bold green]✅ 地图已生成:[/bold green] {out_path}")
        console.print(f"[dim]提示: 在浏览器中打开该文件查看交互式地图[/dim]")

    except Exception as e:
        console.print(f"[bold red]❌ 生成地图失败:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("batch")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), default=Path("./reports"), help="输出目录 (默认: ./reports)")
@_EMPLOYEE_OPT
@_MAPPING_OPT
@click.option("--month", default="", help="仅处理指定月份 (格式: YYYY-MM)")
@click.option("--formats", "-f", default="md,html,csv", help="输出格式组合 (逗号分隔，默认 md,html,csv)")
@click.option("--check-route/--no-check-route", default=True, help="是否调用OSRM对比最短路线")
@click.option("--with-map/--no-map", default=True, help="是否同时生成交互式地图")
@click.option("--split-by-employee/--no-split-by-employee", default=True,
              help="按员工+月份分别生成报告（默认开启，指定 -e 时关闭）")
@click.option("--osrm", default=DEFAULT_OSRM, help="自定义OSRM服务地址")
@_CACHE_OPT
@_POLICY_OPT
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
def batch_cmd(
    input_dir: Path,
    output_dir: Path,
    employee_name: str,
    mapping_csv: Optional[Path],
    month: str,
    formats: str,
    check_route: bool,
    with_map: bool,
    split_by_employee: bool,
    osrm: str,
    cache_dir: Optional[Path],
    policy_config: Optional[Path],
    verbose: bool,
) -> None:
    """批量处理目录下所有GPX文件，按员工和月份聚合生成报告。

    INPUT_DIR: 存放GPX文件的目录 (支持递归查找 *.gpx)

    员工识别优先级:
      1. 命令行 --employee 指定（整批作为同一员工）
      2. --employee-mapping 指定的CSV映射表
      3. 自动从文件名提取（中文2-4字或英文姓名）
      4. 自动从父目录名提取
      5. 兜底为"未知员工"

    输出文件 (按员工+月份分组):
      · {output_dir}/{员工}_YYYY-MM_report.md       Markdown报告
      · {output_dir}/{员工}_YYYY-MM_report.html     HTML报告
      · {output_dir}/{员工}_YYYY-MM_expense.csv     报销CSV
      · {output_dir}/{员工}_YYYY-MM_map.html        轨迹地图
      · {output_dir}/{员工}_YYYY-MM_points.csv      轨迹点CSV

    示例:
      triplog batch ./gpx_files/ --month 2026-06 --employee "张三" -o ./reports
      triplog batch ./gpx/ --mapping employees.csv --formats "html,csv" --no-map
      triplog batch ./gpx/ --split-by-employee --month 2026-06
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolver = _make_resolver(employee_name, str(mapping_csv) if mapping_csv else None, verbose)

        parser = TripParser(verbose=verbose)
        if verbose:
            console.print(f"[dim]扫描目录: {input_dir}[/dim]")
        results = parser.parse_directory(str(input_dir), employee_name=employee_name,
                                         employee_resolver=resolver)

        if not results:
            console.print("[yellow]目录中未找到任何有效GPX文件[/yellow]")
            sys.exit(0)

        if month:
            results = _month_filter(results, month)
            if not results:
                console.print(f"[yellow]月份 {month} 内无任何行程数据[/yellow]")
                sys.exit(0)

        cache_path = str(cache_dir) if cache_dir else str(output_dir / ".osrm_cache")
        reporter = Reporter(
            verbose=verbose, osrm_base=osrm, cache_dir=cache_path,
            policy_config=str(policy_config) if policy_config else None,
        )

        if employee_name:
            split_by_employee = False

        fmt_list = [x.strip().lower() for x in formats.split(",") if x.strip()]
        all_created: List[Path] = []

        def _emit_one(emp: str, ym: str, grp: List[TripResult]) -> None:
            title = f"{emp} {ym} 月度出差行程报告"
            rpt = reporter.build_full_report(grp, employee_name=emp, report_title=title, check_route=check_route)
            safe_emp = emp.replace(" ", "_").replace("/", "_")
            prefix = f"{safe_emp}_{ym}"

            if "md" in fmt_list or "markdown" in fmt_list:
                p = output_dir / f"{prefix}_report.md"
                reporter.save_markdown(rpt, str(p))
                all_created.append(p)
            if "html" in fmt_list:
                p = output_dir / f"{prefix}_report.html"
                reporter.save_html(rpt, str(p))
                all_created.append(p)
            if "pdf" in fmt_list:
                p = output_dir / f"{prefix}_report.pdf"
                pp = reporter.save_pdf(rpt, str(p))
                if pp:
                    all_created.append(Path(pp))
            if "csv" in fmt_list:
                p = output_dir / f"{prefix}_expense.csv"
                reporter.export_csv(rpt, str(p))
                all_created.append(p)
                pp = output_dir / f"{prefix}_points.csv"
                reporter.export_points_csv(grp, str(pp))
                all_created.append(pp)
            if with_map:
                p = output_dir / f"{prefix}_map.html"
                reporter.generate_map(grp, str(p), report=rpt if check_route else None)
                all_created.append(p)

            reporter.print_summary(rpt)

        if split_by_employee:
            groups = _group_by_employee_and_month(results)
            if verbose:
                console.print(f"[dim]按员工+月份分组: {len(groups)} 组[/dim]")
            for (emp, ym), grp in sorted(groups.items()):
                target_ym = ym
                if month and month != ym:
                    continue
                _emit_one(emp, target_ym, grp)
        else:
            if month:
                ym = month
            else:
                all_dates = []
                for r in results:
                    for s in r.segments:
                        if s.start_time:
                            all_dates.append(s.start_time)
                if all_dates:
                    d = min(all_dates)
                    ym = f"{d.year}-{d.month:02d}"
                else:
                    ym = datetime.now().strftime("%Y-%m")

            emp = employee_name or (results[0].employee_name if results else "员工")
            _emit_one(emp, ym, results)

        console.print(f"[bold green]✅ 批量处理完成，共生成 {len(all_created)} 个文件:[/bold green]")
        for f in all_created:
            console.print(f"  📄 {f}")

    except Exception as e:
        console.print(f"[bold red]❌ 批量处理失败:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("export")
@click.argument("gpx_files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dir", "-d", "gpx_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="指定目录批量解析")
@_EMPLOYEE_OPT
@_MAPPING_OPT
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True, help="输出CSV路径")
@click.option("--type", "data_type", type=click.Choice(["expense", "segments", "points"]), default="expense", help="导出类型: expense报销表 / segments行程段 / points全部轨迹点 (默认: expense)")
@click.option("--month", default="", help="仅处理指定月份 (格式: YYYY-MM)")
@click.option("--split-by-employee/--no-split-by-employee", default=False, help="按员工分别导出CSV")
@_POLICY_OPT
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
def export_cmd(
    gpx_files: tuple,
    gpx_dir: Optional[Path],
    employee_name: str,
    mapping_csv: Optional[Path],
    output: Path,
    data_type: str,
    month: str,
    split_by_employee: bool,
    policy_config: Optional[Path],
    verbose: bool,
) -> None:
    """将行程数据导出为CSV，供财务系统直接导入。

    导出类型说明:
      · expense  - 报销汇总表 (每段行程一行，含里程/时长/合规等)
      · segments - 行程段详情表 (同expense)
      · points   - 原始轨迹点表 (每个GPS点一行，含速度/海拔等)

    示例:
      triplog export trip.gpx -o expense.csv --type expense --employee "张三"
      triplog export -d ./gpx/ -o points.csv --type points --month 2026-06
      triplog export -d ./gpx/ -o expense.csv --split-by-employee --mapping emp.csv
    """
    try:
        resolver = _make_resolver(employee_name, str(mapping_csv) if mapping_csv else None, verbose)
        parser = TripParser(verbose=verbose)
        results: List[TripResult] = []

        if gpx_dir:
            results.extend(parser.parse_directory(str(gpx_dir), employee_name=employee_name,
                                                  employee_resolver=resolver))
        for gf in gpx_files:
            emp = employee_name
            if not emp and resolver:
                emp = resolver.resolve(str(gf))
            results.append(parser.parse(str(gf), employee_name=emp))

        if not results:
            console.print("[yellow]未找到任何有效的GPX数据[/yellow]")
            sys.exit(0)

        if month:
            results = _month_filter(results, month)
            if not results:
                console.print(f"[yellow]月份 {month} 内无任何行程数据[/yellow]")
                sys.exit(0)

        reporter = Reporter(
            verbose=verbose,
            policy_config=str(policy_config) if policy_config else None,
        )

        if split_by_employee and not employee_name:
            groups = _group_by_employee_and_month(results)
            for (emp, ym), grp in sorted(groups.items()):
                safe_emp = emp.replace(" ", "_").replace("/", "_")
                base = Path(output)
                per_out = base.with_name(f"{safe_emp}_{ym}_{base.name}")
                rpt = reporter.build_full_report(grp, employee_name=emp, check_route=False)
                if data_type == "points":
                    reporter.export_points_csv(grp, str(per_out))
                else:
                    reporter.export_csv(rpt, str(per_out))
                console.print(f"[green]✅ {emp} CSV:[/green] {per_out}")
            return

        report = reporter.build_full_report(results, employee_name=employee_name, check_route=False)

        if data_type == "points":
            reporter.export_points_csv(results, str(output))
        else:
            reporter.export_csv(report, str(output))

        console.print(f"[bold green]✅ CSV导出成功:[/bold green] {output}")

    except Exception as e:
        console.print(f"[bold red]❌ 导出失败:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _dispatch_output(
    reporter: Reporter,
    report: FullReport,
    results: List[TripResult],
    output: Optional[Path],
    fmt: str,
    csv_path: Optional[Path] = None,
    points_csv: Optional[Path] = None,
) -> None:
    if fmt in ("md", "markdown"):
        out = output or Path("trip_report.md")
        reporter.save_markdown(report, str(out))
    elif fmt == "html":
        out = output or Path("trip_report.html")
        reporter.save_html(report, str(out))
    elif fmt == "pdf":
        out = output or Path("trip_report.pdf")
        reporter.save_pdf(report, str(out))
    else:
        reporter.print_summary(report)

    if csv_path:
        reporter.export_csv(report, str(csv_path))
    if points_csv:
        reporter.export_points_csv(results, str(points_csv))

    if fmt != "console":
        console.print()
        reporter.print_summary(report)


if __name__ == "__main__":
    cli()
