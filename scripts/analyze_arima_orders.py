"""
Scan every ARIMA experiment config, extract the chosen (p,d,q) order from
auto_arima, and tabulate how often each order appears. Frequencies are sliced
by the dimensions that change what an order *means*: time frequency
(1d vs 1h), whether percentage-change was applied, and training period length.

Outputs:
  - console summary
  - LaTeX tables -> tables/arima_orders.tex
  - figure       -> figures/arima_order_distribution.png
"""

import ast
from collections import Counter, defaultdict
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = ROOT / "configs"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
TEX_OUT = TABLES_DIR / "arima_orders.tex"
FIG_OUT = FIGURES_DIR / "arima_order_distribution.pdf"


# ---------- loading ----------

def parse_order(raw):
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return tuple(ast.literal_eval(str(raw)))


def assign_rq(timefreq, exp_group):
    """RQ1 = all 1d configs; RQ2 = 1h non-rq3; RQ3 = 1h rq3-* configs."""
    if timefreq == "1d":
        return "RQ1"
    if exp_group.startswith("rq3-"):
        return "RQ3"
    return "RQ2"


def load_arima_configs():
    rows = []
    for path in CONFIGS_ROOT.rglob("*.yaml"):
        if "/arima/" not in str(path):
            continue
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if not cfg or "arima_order" not in cfg:
            continue
        exp_group = path.parent.name
        timefreq = cfg.get("timefreq")
        rows.append(
            {
                "path": path,
                "exp_group": exp_group,
                "timefreq": timefreq,
                "rq": assign_rq(timefreq, exp_group),
                "pct": bool(cfg.get("percentage_change_applied", False)),
                "diff": bool(cfg.get("differencing_applied", False)),
                "train_value": cfg.get("training_period_value"),
                "train_unit": cfg.get("training_period_unit"),
                "horizon": cfg.get("horizon_length"),
                "order": parse_order(cfg["arima_order"]),
            }
        )
    return rows


# ---------- bucketing ----------

def train_bucket(value, unit):
    """Group training periods into 5 readable buckets."""
    if unit.startswith("d"):
        days = float(value)
    elif unit.startswith("y"):
        days = float(value) * 252
    else:
        return "other"
    if days <= 100:
        return "<=100d"
    if days <= 250:
        return "100-250d"
    if days <= 252 * 3:
        return "1-3y"
    if days <= 252 * 7:
        return "4-7y"
    return "8-10y"


BUCKET_ORDER = ["<=100d", "100-250d", "1-3y", "4-7y", "8-10y"]


# ---------- console helpers ----------

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_dist(label, counter, top=None):
    total = sum(counter.values())
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    print(f"\n[{label}]  n={total}")
    print(f"  {'order':<14}{'count':>8}{'pct':>10}")
    rows = items if top is None else items[:top]
    for order, n in rows:
        print(f"  {str(order):<14}{n:>8}{100*n/total:>9.1f}%")


# ---------- LaTeX helpers ----------

def latex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt_order(o):
    return f"({o[0]},{o[1]},{o[2]})"


RQS = ["RQ1", "RQ2", "RQ3"]


def tex_top_orders_by_rq(rows, pct, top=5):
    """Top-N orders per RQ for one transform (raw or pct), side-by-side."""
    counters = {rq: Counter() for rq in RQS}
    totals = {rq: 0 for rq in RQS}
    for r in rows:
        if r["pct"] != pct:
            continue
        counters[r["rq"]][r["order"]] += 1
        totals[r["rq"]] += 1

    transform_label = "pct\\_change (returns)" if pct else "raw prices"
    label_suffix = "pct" if pct else "raw"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Top-{top} \mbox{{$(p,d,q)$}} orders chosen by \texttt{{auto\_arima}} within each research question, {transform_label}. Within-RQ percentages.}}"
    )
    lines.append(r"\label{tab:arima-orders-by-rq-" + label_suffix + r"}")
    cols = "l" + "rr" * len(RQS)
    lines.append(r"\begin{tabular}{" + cols + r"}")
    lines.append(r"\toprule")
    head = ["order"] + sum([[rf"\multicolumn{{2}}{{c}}{{{rq}}}"] for rq in RQS], [])
    lines.append(" & ".join(head) + r" \\")
    cmid = " ".join(
        [rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(RQS))]
    )
    lines.append(cmid)
    sub = [""] + sum([["count", "\\%"] for _ in RQS], [])
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    all_top = Counter()
    per_rq_top = {}
    for rq in RQS:
        items = sorted(counters[rq].items(), key=lambda kv: -kv[1])[:top]
        per_rq_top[rq] = dict(items)
        for o, n in items:
            all_top[o] += n
    ordered = [o for o, _ in sorted(all_top.items(), key=lambda kv: -kv[1])]

    for order in ordered:
        cells = [fmt_order(order)]
        for rq in RQS:
            n = per_rq_top[rq].get(order)
            if n is None or totals[rq] == 0:
                cells.extend(["--", "--"])
            else:
                cells.extend([str(n), f"{100*n/totals[rq]:.1f}"])
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tex_marginal_d(rows):
    by = defaultdict(Counter)
    for r in rows:
        by[r["pct"]][r["order"][1]] += 1
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Marginal differencing order $d$ chosen by \texttt{auto\_arima}, conditioned on whether percentage-change was applied. Confirms that the transform fully determines $d$: raw prices need integration, returns do not.}")
    lines.append(r"\label{tab:arima-marginal-d}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"transform & $d=0$ & $d=1$ & $d=2$ & $n$ \\")
    lines.append(r"\midrule")
    for pct in [False, True]:
        c = by[pct]
        total = sum(c.values())
        cells = [
            "pct\\_change" if pct else "raw prices",
            f"{c.get(0,0)} ({100*c.get(0,0)/total:.1f}\\%)",
            f"{c.get(1,0)} ({100*c.get(1,0)/total:.1f}\\%)",
            f"{c.get(2,0)} ({100*c.get(2,0)/total:.1f}\\%)",
            str(total),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tex_marginal_pq(rows):
    p_c = Counter(r["order"][0] for r in rows)
    q_c = Counter(r["order"][2] for r in rows)
    n = len(rows)
    keys = sorted(set(list(p_c) + list(q_c)))

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Marginal AR ($p$) and MA ($q$) orders chosen by \texttt{auto\_arima} across all " + str(n) + r" ARIMA configs. The mass on $p=0$ and $q=0$ shows that auto-selection rarely identifies linear lag structure beyond a unit root.}")
    lines.append(r"\label{tab:arima-marginal-pq}")
    lines.append(r"\begin{tabular}{rrrrr}")
    lines.append(r"\toprule")
    lines.append(r"value $k$ & $p=k$ count & \% & $q=k$ count & \% \\")
    lines.append(r"\midrule")
    for k in keys:
        pn, qn = p_c.get(k, 0), q_c.get(k, 0)
        lines.append(
            f"{k} & {pn} & {100*pn/n:.1f} & {qn} & {100*qn/n:.1f}" + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tex_orders_by_train_bucket(rows, rq, pct=False, top=4):
    """Top orders per training-period bucket, restricted to one (RQ, transform) cell."""
    counters = {b: Counter() for b in BUCKET_ORDER}
    totals = {b: 0 for b in BUCKET_ORDER}
    for r in rows:
        if r["rq"] != rq or r["pct"] != pct:
            continue
        b = train_bucket(r["train_value"], r["train_unit"])
        if b in counters:
            counters[b][r["order"]] += 1
            totals[b] += 1
    buckets = [b for b in BUCKET_ORDER if totals[b] > 0]

    transform = "pct\\_change" if pct else "raw prices"
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Top-{top} \mbox{{$(p,d,q)$}} orders by training-period length, {rq} {transform}. Order complexity grows only with multi-year training windows.}}")
    lines.append(r"\label{tab:arima-orders-by-train-" + rq.lower() + ("-pct" if pct else "-raw") + "}")
    cols = "l" + "rr" * len(buckets)
    lines.append(r"\begin{tabular}{" + cols + r"}")
    lines.append(r"\toprule")
    head = ["order"] + sum([[rf"\multicolumn{{2}}{{c}}{{{b}}}"] for b in buckets], [])
    lines.append(" & ".join(head) + r" \\")
    cmid = " ".join([rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(buckets))])
    lines.append(cmid)
    sub = [""] + sum([[f"n={totals[b]}", ""] for b in buckets], [])
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")
    sub = [""] + sum([["count", "\\%"] for _ in buckets], [])
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    all_top = Counter()
    per_bucket_top = {}
    for b in buckets:
        items = sorted(counters[b].items(), key=lambda kv: -kv[1])[:top]
        per_bucket_top[b] = dict(items)
        for o, n in items:
            all_top[o] += n
    ordered = [o for o, _ in sorted(all_top.items(), key=lambda kv: -kv[1])]

    for order in ordered:
        cells = [fmt_order(order)]
        for b in buckets:
            n = per_bucket_top[b].get(order)
            if n is None:
                cells.extend(["--", "--"])
            else:
                cells.extend([str(n), f"{100*n/totals[b]:.1f}"])
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------- figure ----------

def make_figure(rows, out_path):
    transforms = [False, True]  # rows: raw, pct
    buckets = [(rq, pct) for pct in transforms for rq in RQS]
    counters = {b: Counter() for b in buckets}
    for r in rows:
        key = (r["rq"], r["pct"])
        if key in counters:
            counters[key][r["order"]] += 1

    # Stable colour per order: union of top-5 across all panels.
    union, seen = [], set()
    for b in buckets:
        for order, _ in sorted(counters[b].items(), key=lambda kv: -kv[1])[:5]:
            if order not in seen:
                union.append(order)
                seen.add(order)
    cmap = plt.get_cmap("tab20")
    colors = {o: cmap(i % 20) for i, o in enumerate(union)}

    rq_subtitles = {
        "RQ1": "Daily (1d)",
        "RQ2": "Hourly, normal periods (1h)",
        "RQ3": "Hourly, crisis periods (1h)",
    }
    row_labels = {False: "Raw prices", True: "Returns (pct change)"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for row_idx, pct in enumerate(transforms):
        for col_idx, rq in enumerate(RQS):
            ax = axes[row_idx][col_idx]
            c = counters[(rq, pct)]
            total = sum(c.values()) or 1
            items = sorted(c.items(), key=lambda kv: -kv[1])[:5]
            labels = [fmt_order(o) for o, _ in items]
            pcts = [100 * n / total for _, n in items]
            bar_colors = [colors.get(o, "lightgray") for o, _ in items]
            bars = ax.bar(labels, pcts, color=bar_colors, edgecolor="black", linewidth=0.5)
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=0)
            for bar, p in zip(bars, pcts):
                ax.text(bar.get_x() + bar.get_width() / 2, p + 2, f"{p:.0f}%",
                        ha="center", va="bottom", fontsize=9)

            # Column header only on top row.
            if row_idx == 0:
                ax.set_title(f"{rq}\n{rq_subtitles[rq]}", fontsize=11)

        # Row label on the left-most panel.
        axes[row_idx][0].set_ylabel(f"{row_labels[pct]}\n% of configs", fontsize=11)

    fig.suptitle(
        "Distribution of ARIMA $(p,d,q)$ orders selected by auto_arima",
        fontsize=14, y=0.99,
    )
    fig.text(
        0.5, 0.935,
        "Top-5 orders per research question and data transform; percentages computed within each panel",
        ha="center", fontsize=10, style="italic", color="dimgray",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure written: {out_path}")


# ---------- main ----------

def main():
    rows = load_arima_configs()
    print(f"Loaded {len(rows)} ARIMA configs from {CONFIGS_ROOT}")

    # console summaries (kept short)
    overall = Counter(r["order"] for r in rows)
    header("Overall (p,d,q) distribution")
    print_dist("ALL", overall, top=10)

    # tables
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    parts = [
        r"% Auto-generated by scripts/analyze_arima_orders.py",
        r"% Tables summarising auto_arima order selection across all ARIMA configs.",
        "",
        tex_top_orders_by_rq(rows, pct=False, top=5),
        "",
        tex_top_orders_by_rq(rows, pct=True, top=5),
        "",
        tex_marginal_d(rows),
        "",
        tex_marginal_pq(rows),
        "",
        tex_orders_by_train_bucket(rows, rq="RQ1", pct=False, top=4),
        "",
        tex_orders_by_train_bucket(rows, rq="RQ1", pct=True, top=4),
        "",
        tex_orders_by_train_bucket(rows, rq="RQ2", pct=False, top=4),
        "",
        tex_orders_by_train_bucket(rows, rq="RQ2", pct=True, top=4),
        "",
    ]
    TEX_OUT.write_text("\n".join(parts))
    print(f"\nLaTeX tables written: {TEX_OUT}")

    # figure
    make_figure(rows, FIG_OUT)


if __name__ == "__main__":
    main()
