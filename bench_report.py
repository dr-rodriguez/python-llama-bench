#!/usr/bin/env python3
"""
bench_report.py — Generate an HTML benchmark report from llama_bench JSON results.

Usage:
    python bench_report.py                        # reads ./data/*.json
    python bench_report.py --data ./results       # custom data directory
    python bench_report.py --output report.html   # custom output file
    python bench_report.py data/a.json data/b.json  # explicit files
"""

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def load_files(paths: list[Path]) -> list[dict]:
    models = []
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            # Normalise: ensure top-level fields exist
            data.setdefault("model", p.stem)
            data.setdefault("timestamp", "")
            data.setdefault("score", {"passed": 0, "total": 0})
            data.setdefault("results", [])
            data["_source"] = str(p)
            models.append(data)
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}", file=sys.stderr)
    return sorted(models, key=lambda d: d["model"])


def aggregate(model: dict) -> dict:
    """Compute per-model aggregate stats."""
    results = model["results"]
    if not results:
        return {}

    def mvals(key):
        return [r["metrics"][key] for r in results if r["metrics"].get(key, 0) > 0]

    def med(lst):
        return statistics.median(lst) if lst else 0.0

    categories = sorted(set(r["category"] for r in results))
    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_stats[cat] = {
            "passed": sum(1 for r in cat_results if r["passed"]),
            "total": len(cat_results),
        }

    pp_speeds = mvals("prompt_speed_tok_s")
    tg_speeds = mvals("gen_speed_tok_s")
    wall_times = mvals("total_wall_time_ms")

    return {
        "pass_rate": model["score"]["passed"] / model["score"]["total"] * 100
            if model["score"]["total"] else 0,
        "median_pp":   med(pp_speeds),
        "median_tg":   med(tg_speeds),
        "median_wall": med(wall_times),
        "p95_wall":    sorted(wall_times)[int(len(wall_times) * 0.95)] if wall_times else 0,
        "total_wall":  sum(mvals("total_wall_time_ms")),
        "cat_stats":   cat_stats,
        "pp_speeds":   pp_speeds,
        "tg_speeds":   tg_speeds,
        "wall_times":  wall_times,
    }


# ---------------------------------------------------------------------------
# Chart helpers (inline SVG — no dependencies)
# ---------------------------------------------------------------------------

def spark_bar(value: float, max_val: float, width: int = 120, height: int = 18,
              color: str = "#00d4aa") -> str:
    w = int((value / max_val) * width) if max_val else 0
    return (
        f'<svg width="{width}" height="{height}" class="spark">'
        f'<rect x="0" y="4" width="{width}" height="{height-8}" rx="2" fill="#1e2533"/>'
        f'<rect x="0" y="4" width="{w}" height="{height-8}" rx="2" fill="{color}"/>'
        f'</svg>'
    )


def donut(passed: int, total: int, size: int = 80) -> str:
    if total == 0:
        return ""
    pct = passed / total
    r = (size - 12) / 2
    cx = cy = size / 2
    circ = 2 * math.pi * r
    arc = pct * circ
    gap = circ - arc
    # Color: green ≥80%, amber ≥60%, red otherwise
    color = "#00d4aa" if pct >= 0.8 else "#f59e0b" if pct >= 0.6 else "#ef4444"
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="donut">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e2533" stroke-width="8"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="8"
    stroke-dasharray="{arc:.2f} {gap:.2f}"
    stroke-dashoffset="{circ/4:.2f}"
    stroke-linecap="round"/>
  <text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle"
    font-family="'JetBrains Mono',monospace" font-size="13" font-weight="700" fill="{color}">
    {int(pct*100)}%
  </text>
</svg>"""


def bar_chart(
    series: list[tuple[str, list[tuple[float, float]]]],
    labels: list[str],
    ylabel: str,
    height: int = 180,
    bar_w: int = 22,
    gap: int = 8,
) -> str:
    """
    Grouped vertical bar chart with optional error bars.

    series: list of (name, [(median, stdev), ...])
            stdev=0 draws no error bar.
    labels: x-axis group labels (one per group).
    """
    COLORS = [
        "#00d4aa",  # teal
        "#818cf8",  # indigo
        "#f59e0b",  # amber
        "#f472b6",  # pink
        "#34d399",  # emerald
        "#60a5fa",  # sky blue
        "#fb923c",  # orange
        "#a78bfa",  # violet
        "#4ade80",  # green
        "#f87171",  # coral red
    ]

    # Estimate legend layout: each item is swatch(10) + gap(4) + ~8px/char text
    LEGEND_ITEM_W = 160          # px reserved per legend item in a row
    LEGEND_ROW_H  = 18           # px per legend row
    ITEMS_PER_ROW = max(1, 520 // LEGEND_ITEM_W)   # how many fit per row
    legend_rows   = math.ceil(len(series) / ITEMS_PER_ROW)
    LEGEND_H      = legend_rows * LEGEND_ROW_H + 6  # total legend height

    X_LABEL_H = 24
    PAD_L     = 50
    PAD_TOP   = LEGEND_H + 8
    plot_h    = height - PAD_TOP - X_LABEL_H

    n_groups  = len(labels)
    n_series  = len(series)
    group_w   = n_series * (bar_w + 2) + gap
    bar_area_w = n_groups * group_w + PAD_L + 10

    # Legend needs enough width for ITEMS_PER_ROW items
    legend_area_w = PAD_L + ITEMS_PER_ROW * LEGEND_ITEM_W
    total_w = max(bar_area_w, legend_area_w)
    total_h = height

    all_medians = [pt[0] for _, pts in series for pt in pts]
    all_stdevs  = [pt[1] for _, pts in series for pt in pts]
    max_val = max((m + s) for m, s in zip(all_medians, all_stdevs)) * 1.12 if all_medians else 1

    S = []
    S.append(
        f'<svg width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}" '
        f'class="bar-chart" font-family="\'Inter\',sans-serif" font-size="10">'
    )

    # ── Legend (wraps into rows) ──────────────────────────────────────────
    for si, (sname, _) in enumerate(series):
        color  = COLORS[si % len(COLORS)]
        row    = si // ITEMS_PER_ROW
        col    = si %  ITEMS_PER_ROW
        lx     = PAD_L + col * LEGEND_ITEM_W
        ly_top = row * LEGEND_ROW_H + 4
        # Swatch
        S.append(f'<rect x="{lx}" y="{ly_top}" width="10" height="10" rx="2" fill="{color}"/>')
        # Label — truncate long names with ellipsis to fit within the reserved slot
        display = sname if len(sname) <= 22 else sname[:20] + "…"
        S.append(
            f'<text x="{lx+14}" y="{ly_top+9}" fill="#8892aa" font-size="10"'
            f' dominant-baseline="auto">{display}</text>'
        )
        # Full name in title for hover tooltip
        if display != sname:
            S.append(f'<title>{sname}</title>')

    # ── Y-axis gridlines ──────────────────────────────────────────────────
    for i in range(5):
        y   = PAD_TOP + plot_h * i / 4
        val = max_val * (1 - i / 4)
        S.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{total_w-5}" y2="{y:.1f}" '
            f'stroke="#1e2533" stroke-width="1"/>'
            f'<text x="{PAD_L-4}" y="{y+3:.1f}" text-anchor="end" fill="#6b7a99">{val:.0f}</text>'
        )

    # ── Bars + error bars ─────────────────────────────────────────────────
    for gi, label in enumerate(labels):
        gx = PAD_L + 4 + gi * group_w
        for si, (sname, pts) in enumerate(series):
            median_val, stdev_val = pts[gi] if gi < len(pts) else (0, 0)
            bh    = int((median_val / max_val) * plot_h) if max_val else 0
            bx    = gx + si * (bar_w + 2)
            by    = PAD_TOP + plot_h - bh
            color = COLORS[si % len(COLORS)]
            cx    = bx + bar_w / 2

            # Bar
            S.append(
                f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" '
                f'rx="3" fill="{color}" opacity="0.85">'
                f'<title>{sname}: {median_val:.1f} ± {stdev_val:.1f} tok/s</title></rect>'
            )

            # Error bar (only when stdev > 0 and bar is tall enough to be meaningful)
            if stdev_val > 0 and bh > 4:
                sd_px    = (stdev_val / max_val) * plot_h  if max_val else 0
                cap_w    = max(4, bar_w // 4)
                top_y    = by - sd_px
                bot_y    = by + min(sd_px, bh - 2)   # don't extend below bar base
                # Vertical spine
                S.append(
                    f'<line x1="{cx:.1f}" y1="{top_y:.1f}" x2="{cx:.1f}" y2="{bot_y:.1f}" '
                    f'stroke="white" stroke-width="1.5" opacity="0.7"/>'
                )
                # Top cap
                S.append(
                    f'<line x1="{cx - cap_w/2:.1f}" y1="{top_y:.1f}" '
                    f'x2="{cx + cap_w/2:.1f}" y2="{top_y:.1f}" '
                    f'stroke="white" stroke-width="1.5" opacity="0.7"/>'
                )
                # Bottom cap (only if it's visible above the bar base)
                if bot_y < PAD_TOP + plot_h - 1:
                    S.append(
                        f'<line x1="{cx - cap_w/2:.1f}" y1="{bot_y:.1f}" '
                        f'x2="{cx + cap_w/2:.1f}" y2="{bot_y:.1f}" '
                        f'stroke="white" stroke-width="1.5" opacity="0.7"/>'
                    )

            # Value label above bar (includes ± stdev if present)
            if bh > 22:
                label_txt = f"{median_val:.0f}"
                if stdev_val > 0:
                    label_txt += f" ±{stdev_val:.0f}"
                S.append(
                    f'<text x="{cx:.1f}" y="{by - (sd_px + 5 if stdev_val > 0 else 3):.1f}" '
                    f'text-anchor="middle" fill="{color}" font-size="9">{label_txt}</text>'
                )

        # X-axis group label
        label_x = gx + (n_series * (bar_w + 2)) / 2 - 1
        label_y = PAD_TOP + plot_h + X_LABEL_H - 6
        if label:
            S.append(
                f'<text x="{label_x:.1f}" y="{label_y}" text-anchor="middle" '
                f'fill="#8892aa" font-size="10">{label}</text>'
            )

    # ── Y-axis rotated label ──────────────────────────────────────────────
    mid_y = PAD_TOP + plot_h / 2
    S.append(
        f'<text x="10" y="{mid_y:.0f}" text-anchor="middle" fill="#6b7a99" '
        f'font-size="10" transform="rotate(-90,10,{mid_y:.0f})">{ylabel}</text>'
    )

    S.append("</svg>")
    return "\n".join(S)


def scatter_svg(points: list[tuple[float, float, str]], xlabel: str, ylabel: str,
                width: int = 400, height: int = 220) -> str:
    """Wall time vs gen token count scatter."""
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = max(xs) * 1.1, max(ys) * 1.1
    pad_l, pad_b = 52, 36

    def px(x): return pad_l + (x / mx) * (width - pad_l - 10)
    def py(y): return (height - pad_b) - (y / my) * (height - pad_b - 10)

    lines = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="\'Inter\',sans-serif" font-size="10">'
    ]
    # Grid
    for i in range(5):
        gx = pad_l + i * (width - pad_l - 10) / 4
        gy = 10 + i * (height - pad_b - 10) / 4
        lines.append(
            f'<line x1="{gx:.0f}" y1="10" x2="{gx:.0f}" y2="{height-pad_b}" '
            f'stroke="#1e2533" stroke-width="1"/>'
            f'<text x="{gx:.0f}" y="{height-pad_b+12}" text-anchor="middle" fill="#6b7a99">'
            f'{mx * i / 4:.0f}</text>'
        )
        lines.append(
            f'<line x1="{pad_l}" y1="{gy:.0f}" x2="{width-10}" y2="{gy:.0f}" '
            f'stroke="#1e2533" stroke-width="1"/>'
            f'<text x="{pad_l-4}" y="{gy+3:.0f}" text-anchor="end" fill="#6b7a99">'
            f'{my * (1 - i/4):.0f}</text>'
        )
    # Points
    COLORS = ["#00d4aa", "#818cf8", "#f59e0b", "#ef4444", "#34d399"]
    cat_colors = {}
    ci = 0
    for x, y, label in points:
        cat = label.split(":")[0] if ":" in label else label
        if cat not in cat_colors:
            cat_colors[cat] = COLORS[ci % len(COLORS)]
            ci += 1
        color = cat_colors[cat]
        lines.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="{color}" '
            f'opacity="0.8"><title>{label}</title></circle>'
        )
    # Axes labels (each appended separately — never concatenate two SVG elements into one call)
    lines.append(
        f'<text x="{(width + pad_l) // 2}" y="{height - 2}" text-anchor="middle" '
        f'fill="#8892aa" font-size="10">{xlabel}</text>'
    )
    mid_y = (height - pad_b) // 2
    lines.append(
        f'<text x="10" y="{mid_y}" text-anchor="middle" fill="#8892aa" '
        f'font-size="10" transform="rotate(-90,10,{mid_y})">{ylabel}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def render_html(models: list[dict], aggs: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(models)

    # ── Summary cards ──────────────────────────────────────────────────────
    cards_html = []
    for m, ag in zip(models, aggs):
        passed = m["score"]["passed"]
        total  = m["score"]["total"]
        cards_html.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="model-name">{m['model']}</span>
          </div>
          <div class="card-body">
            <div class="donut-wrap">
              {donut(passed, total, 88)}
              <div class="score-label">{passed}/{total} passed</div>
            </div>
            <div class="card-stats">
              <div class="stat"><span class="stat-label">PP speed</span>
                <span class="stat-val">{ag['median_pp']:.0f} <span class="stat-unit">tok/s</span></span></div>
              <div class="stat"><span class="stat-label">TG speed</span>
                <span class="stat-val">{ag['median_tg']:.1f} <span class="stat-unit">tok/s</span></span></div>
              <div class="stat"><span class="stat-label">Median wall</span>
                <span class="stat-val">{fmt_ms(ag['median_wall'])}</span></div>
              <div class="stat"><span class="stat-label">Total time</span>
                <span class="stat-val">{fmt_ms(ag['total_wall'])}</span></div>
            </div>
          </div>
        </div>""")

    # ── Per-exercise table (all models side by side) ───────────────────────
    # Collect all exercise IDs/descriptions in order
    all_exercises: dict[int, dict] = {}
    for m in models:
        for r in m["results"]:
            if r["exercise_id"] not in all_exercises:
                all_exercises[r["exercise_id"]] = {
                    "id": r["exercise_id"],
                    "category": r["category"],
                    "description": r["description"],
                }
    exercises = sorted(all_exercises.values(), key=lambda x: x["id"])

    # Build lookup: model_name -> exercise_id -> result
    lookup: dict[str, dict[int, dict]] = {}
    for m in models:
        lookup[m["model"]] = {r["exercise_id"]: r for r in m["results"]}

    CAT_COLORS = {
        "Math":        "#06b6d4",
        "Factual":     "#818cf8",
        "Reasoning":   "#c084fc",
        "Instruction": "#f59e0b",
        "Code":        "#34d399",
        "Extraction":  "#f87171",
    }

    model_headers = "".join(
        f'<th class="model-col">{m["model"]}</th>' for m in models
    )

    rows = []
    prev_cat = None
    for ex in exercises:
        cat = ex["category"]
        if cat != prev_cat:
            rows.append(
                f'<tr class="cat-row"><td colspan="{3+n}" style="color:{CAT_COLORS.get(cat,"#aaa")}">'
                f'<span class="cat-label">▸ {cat}</span></td></tr>'
            )
            prev_cat = cat

        cells = []
        for m in models:
            res = lookup[m["model"]].get(ex["id"])
            if res is None:
                cells.append('<td class="no-data">—</td>')
            elif res["passed"]:
                tg = res["metrics"]["gen_speed_tok_s"]
                wall = res["metrics"]["total_wall_time_ms"]
                cells.append(
                    f'<td class="pass">'
                    f'<span class="badge pass-badge">✓</span>'
                    f'<span class="cell-metrics">{tg:.1f} tok/s · {fmt_ms(wall)}</span>'
                    f'</td>'
                )
            else:
                err = (res.get("error") or "")[:40]
                cells.append(
                    f'<td class="fail">'
                    f'<span class="badge fail-badge">✗</span>'
                    f'<span class="cell-metrics">{err}</span>'
                    f'</td>'
                )

        row_class = "dim-row" if ex["id"] % 2 == 0 else ""
        rows.append(
            f'<tr class="{row_class}">'
            f'<td class="ex-id">{ex["id"]:02d}</td>'
            f'<td class="ex-desc">{ex["description"]}</td>'
            f'{"".join(cells)}</tr>'
        )

    table_html = f"""
    <table class="results-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Exercise</th>
          {model_headers}
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>"""

    # ── Speed comparison charts ───────────────────────────────────────────
    # One bar per model, averaged across all exercises, with stdev error bar.
    # series = [(model_name, [(median, stdev)]), ...]
    categories = sorted(set(ex["category"] for ex in exercises))
    tg_series = []
    pp_series = []
    for m in models:
        tg_v = [r["metrics"]["gen_speed_tok_s"] for r in m["results"] if r["metrics"]["gen_speed_tok_s"] > 0]
        pp_v = [r["metrics"]["prompt_speed_tok_s"] for r in m["results"] if r["metrics"]["prompt_speed_tok_s"] > 0]
        tg_med = statistics.median(tg_v) if tg_v else 0
        pp_med = statistics.median(pp_v) if pp_v else 0
        tg_sd  = statistics.stdev(tg_v)  if len(tg_v) > 1 else 0
        pp_sd  = statistics.stdev(pp_v)  if len(pp_v) > 1 else 0
        tg_series.append((m["model"], [(tg_med, tg_sd)]))
        pp_series.append((m["model"], [(pp_med, pp_sd)]))

    # bar_w scales so the chart fills roughly 480px regardless of model count
    _bar_w = max(28, min(80, 480 // max(n, 1)))
    tg_chart = bar_chart(tg_series, [""], "tok/s", height=220, bar_w=_bar_w)
    pp_chart = bar_chart(pp_series, [""], "tok/s", height=220, bar_w=_bar_w)

    # ── Category accuracy table ────────────────────────────────────────────
    cat_table_rows = []
    for cat in categories:
        color = CAT_COLORS.get(cat, "#aaa")
        cells = [f'<td style="color:{color};font-weight:600">{cat}</td>']
        for m, ag in zip(models, aggs):
            cs = ag["cat_stats"].get(cat, {"passed": 0, "total": 0})
            pct = cs["passed"] / cs["total"] * 100 if cs["total"] else 0
            bar = spark_bar(pct, 100, width=80, color=color)
            pct_color = "#00d4aa" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
            cells.append(
                f'<td><div class="cat-cell">'
                f'{bar}'
                f'<span style="color:{pct_color}">{cs["passed"]}/{cs["total"]}</span>'
                f'</div></td>'
            )
        cat_table_rows.append(f'<tr>{"".join(cells)}</tr>')

    cat_model_headers = "".join(f'<th>{m["model"]}</th>' for m in models)
    cat_table_html = f"""
    <table class="cat-table">
      <thead><tr><th>Category</th>{cat_model_headers}</tr></thead>
      <tbody>{"".join(cat_table_rows)}</tbody>
    </table>"""

    # ── Speed stats table ──────────────────────────────────────────────────
    speed_rows = []
    for m, ag in zip(models, aggs):
        pp_vals = ag["pp_speeds"]
        tg_vals = ag["tg_speeds"]
        def p95(lst):
            return sorted(lst)[int(len(lst)*0.95)] if lst else 0
        speed_rows.append(f"""
        <tr>
          <td class="model-name-cell">{m['model']}</td>
          <td>{ag['median_pp']:.0f}</td>
          <td>{(max(pp_vals) if pp_vals else 0):.0f}</td>
          <td>{(min(pp_vals) if pp_vals else 0):.0f}</td>
          <td>{ag['median_tg']:.1f}</td>
          <td>{(max(tg_vals) if tg_vals else 0):.1f}</td>
          <td>{(min(tg_vals) if tg_vals else 0):.1f}</td>
          <td>{fmt_ms(ag['median_wall'])}</td>
          <td>{fmt_ms(ag['p95_wall'])}</td>
          <td>{fmt_ms(ag['total_wall'])}</td>
        </tr>""")

    speed_table_html = f"""
    <table class="speed-table">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="3">Prompt Processing (tok/s)</th>
          <th colspan="3">Token Generation (tok/s)</th>
          <th colspan="3">Wall Time</th>
        </tr>
        <tr>
          <th>Median</th><th>Max</th><th>Min</th>
          <th>Median</th><th>Max</th><th>Min</th>
          <th>Median</th><th>p95</th><th>Total</th>
        </tr>
      </thead>
      <tbody>{"".join(speed_rows)}</tbody>
    </table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llama.cpp Benchmark Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #0d1117;
    --surface: #161b27;
    --border:  #1e2533;
    --border2: #252f42;
    --text:    #c9d1e0;
    --muted:   #6b7a99;
    --accent:  #00d4aa;
    --accent2: #818cf8;
    --pass:    #00d4aa;
    --fail:    #ef4444;
    --warn:    #f59e0b;
    --font-mono: 'JetBrains Mono', monospace;
    --font:      'Inter', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── Layout ── */
  header {{
    border-bottom: 1px solid var(--border);
    padding: 24px 40px;
    display: flex;
    align-items: baseline;
    gap: 16px;
    position: sticky; top: 0;
    background: var(--bg);
    z-index: 10;
  }}
  header h1 {{
    font-family: var(--font-mono);
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }}
  header .subtitle {{
    font-size: 12px;
    color: var(--muted);
    font-family: var(--font-mono);
  }}
  nav {{
    margin-left: auto;
    display: flex;
    gap: 20px;
  }}
  nav a {{
    color: var(--muted);
    text-decoration: none;
    font-size: 12px;
    font-family: var(--font-mono);
    transition: color .15s;
  }}
  nav a:hover {{ color: var(--accent); }}

  main {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 36px 40px 80px;
  }}
  section {{ margin-bottom: 52px; scroll-margin-top: 80px; }}
  section + section {{ border-top: 1px solid var(--border); padding-top: 44px; }}

  h2 {{
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
  }}

  h2 span {{ color: var(--accent); margin-right: 8px; }}

  /* ── Cards ── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    transition: border-color .2s;
  }}
  .card:hover {{ border-color: var(--border2); }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 16px;
    gap: 8px;
  }}
  .model-name {{
    font-family: var(--font-mono);
    font-size: 15px;
    font-weight: 700;
    color: #e2e8f0;
    word-break: break-all;
  }}

  .card-body {{
    display: flex;
    gap: 20px;
    align-items: center;
  }}
  .donut-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
  }}
  .score-label {{ font-size: 11px; color: var(--muted); font-family: var(--font-mono); }}
  .card-stats {{ flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; }}
  .stat {{ display: flex; flex-direction: column; }}
  .stat-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }}
  .stat-val {{
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.2;
  }}
  .stat-unit {{ font-size: 10px; color: var(--muted); font-weight: 400; }}

  /* ── Tables ── */
  /* overflow-x: clip lets sticky positioning work (unlike auto/scroll) */
  .table-wrap {{ overflow-x: clip; }}
  /* Fallback for browsers that don't support clip: the table is still readable */
  @supports not (overflow-x: clip) {{
    .table-wrap {{ overflow-x: auto; }}
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  .speed-table {{ min-width: 700px; width: auto; }}
  th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  th {{
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .5px;
    background: var(--surface);
  }}
  /* Sticky headers — only works when table is not inside overflow:auto wrapper.
     We give each table its own scrollable wrapper so sticky can work. */
  /* Headers stick to the top of their scroll-container wrapper (top:0,
     not 72px — these wrappers are scroll containers, so the offset is
     relative to the wrapper, not the page). */
  .results-table th {{
    position: sticky;
    top: 0;
  }}
  .cat-table th {{
    position: sticky;
    top: 0;
  }}
  .speed-table thead tr:first-child th {{
    position: sticky;
    top: 72px;
  }}
  /* Speed table second sub-header row — sits ~34px below the first row */
  .speed-table thead tr:nth-child(2) th {{
    position: sticky;
    top: 106px;
  }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}

  .results-table .ex-id {{
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
    width: 32px;
  }}
  .results-table .ex-desc {{ color: var(--text); max-width: 260px; white-space: normal; }}
  .results-table .cat-row td {{
    font-size: 11px;
    font-family: var(--font-mono);
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 14px 12px 6px;
    background: var(--bg);
    border-bottom: none;
  }}
  .results-table .dim-row td {{ background: rgba(255,255,255,0.01); }}
  .results-table .pass {{ color: var(--pass); }}
  .results-table .fail {{ color: var(--fail); }}
  .results-table .no-data {{ color: var(--muted); }}
  .badge {{
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    margin-right: 6px;
  }}
  .pass-badge {{ color: var(--pass); }}
  .fail-badge {{ color: var(--fail); }}
  .cell-metrics {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
  }}
  .model-col {{ min-width: 180px; }}

  /* ── Horizontally scrollable tables with pinned left column(s) ──
     overflow-x:auto makes the wrapper a scroll container so wide tables
     (many models) can scroll left/right. The left column(s) stay pinned
     via position:sticky; left. */
  .scroll-x {{ overflow-x: auto; }}

  /* Accuracy by Category — pin the Category column */
  .cat-table th:first-child,
  .cat-table td:first-child {{
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--surface);
  }}
  .cat-table thead th:first-child {{ z-index: 3; }}

  /* Exercise Detail — pin the # and Exercise columns */
  .results-table th:nth-child(1),
  .results-table td.ex-id {{
    position: sticky;
    left: 0;
    z-index: 1;
    width: 44px;
    min-width: 44px;
    background: var(--bg);
  }}
  .results-table th:nth-child(2),
  .results-table td.ex-desc {{
    position: sticky;
    left: 44px;
    z-index: 1;
    min-width: 200px;
    background: var(--bg);
  }}
  .results-table thead th:nth-child(1),
  .results-table thead th:nth-child(2) {{ z-index: 3; background: var(--surface); }}
  /* Category separator row — sticky on a colspan <td> is unreliable, so
     pin an inner inline-block span to the wrapper's left edge instead. */
  .results-table .cat-label {{
    position: sticky;
    left: 12px;
    display: inline-block;
  }}

  /* Category accuracy table */
  .cat-table td, .cat-table th {{ padding: 10px 16px; }}
  .cat-cell {{ display: flex; align-items: center; gap: 12px; }}
  .cat-cell span {{ font-family: var(--font-mono); font-size: 12px; font-weight: 700; }}
  .spark {{ display: block; }}

  /* Speed table */
  .speed-table th {{ text-align: center; }}
  .speed-table td {{ text-align: right; font-family: var(--font-mono); font-size: 13px; }}
  .speed-table .model-name-cell {{ text-align: left; font-weight: 600; color: #e2e8f0; }}
  .speed-table thead tr:first-child th {{
    background: var(--surface);
    border-bottom: 1px solid var(--border2);
  }}
  .speed-table thead tr:nth-child(2) th {{
    background: #111622;
    font-size: 10px;
    color: var(--muted);
  }}
  /* The rowspan="2" Model cell covers both header rows */
  .speed-table th[rowspan] {{
    vertical-align: middle;
    background: var(--surface);
    z-index: 2;
  }}

  /* Charts */
  .charts-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  @media (max-width: 900px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}
  .chart-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    overflow-x: auto;
  }}
  .chart-title {{
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
  }}
  .bar-chart {{ display: block; max-width: 100%; }}
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    font-family: var(--font-mono);
    padding: 24px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <h1>⚡ llama.cpp bench</h1>
  <span class="subtitle">{n} model{'s' if n != 1 else ''} · generated {now}</span>
  <nav>
    <a href="#summary">Summary</a>
    <a href="#accuracy">Accuracy</a>
    <a href="#speed">Speed</a>
    <a href="#exercises">Exercises</a>
  </nav>
</header>

<main>

  <section id="summary">
    <h2><span>01</span>Model Summary</h2>
    <div class="cards">
      {"".join(cards_html)}
    </div>
  </section>

  <section id="accuracy">
    <h2><span>02</span>Accuracy by Category</h2>
    <div class="table-wrap scroll-x">{cat_table_html}</div>
  </section>

  <section id="speed">
    <h2><span>03</span>Speed Stats</h2>
    <div class="table-wrap">{speed_table_html}</div>

    <div class="charts-row" style="margin-top:24px">
      <div class="chart-box">
        <div class="chart-title">Token Generation Speed by Category (tok/s)</div>
        {tg_chart}
      </div>
      <div class="chart-box">
        <div class="chart-title">Prompt Processing Speed by Category (tok/s)</div>
        {pp_chart}
      </div>
    </div>
  </section>

  <section id="exercises">
    <h2><span>04</span>Exercise Detail</h2>
    <div class="table-wrap scroll-x">{table_html}</div>
  </section>

</main>

<footer>llama.cpp bench report · {now}</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML benchmark report from llama_bench JSON files."
    )
    parser.add_argument(
        "files", nargs="*",
        help="JSON result files (default: all *.json in --data directory)"
    )
    parser.add_argument(
        "--data", default="data",
        help="Directory to scan for *.json files (default: ./data)"
    )
    parser.add_argument(
        "--output", default="bench_report.html",
        help="Output HTML file (default: bench_report.html)"
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        data_dir = Path(args.data)
        if not data_dir.exists():
            print(f"[error] Data directory '{data_dir}' not found. "
                  "Pass JSON files as arguments or use --data.", file=sys.stderr)
            sys.exit(1)
        paths = sorted(data_dir.glob("*.json"))
        if not paths:
            print(f"[error] No *.json files found in '{data_dir}'.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {len(paths)} file(s)...")
    models = load_files(paths)
    if not models:
        print("[error] No valid JSON files could be loaded.", file=sys.stderr)
        sys.exit(1)

    aggs = [aggregate(m) for m in models]

    print(f"Generating report for: {', '.join(m['model'] for m in models)}")
    html = render_html(models, aggs)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Report written to: {out.resolve()}")


if __name__ == "__main__":
    main()