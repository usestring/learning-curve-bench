"""Metrics over records.jsonl files. Every headline number uses first attempts only.

A first attempt happens before any feedback in that episode, so it measures what the agent
carried in from earlier episodes. The retry after feedback is reported separately
("feedback uptake") and never mixed into the learning numbers. CLE's design mixes the two:
there, feedback comes with an extra attempt.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .families import BASE

CONTROL = "log_count"


def load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_metrics(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["index"])
    learnable = [r for r in rows if r["family"] != CONTROL]
    m: dict = {
        "episodes": len(rows),
        "first_try_success": _mean([r["solved1"] for r in learnable]),
        "aulc": _mean([r["score1"] for r in learnable]),
        "post_first_success": _mean([r["solved1"] for r in learnable if r["occurrence"] > 0]),
        "probe_success": _mean([r["solved1"] for r in learnable if r["probe"]]),
        "feedback_uptake": _mean([r["solved2"] for r in rows if r["solved2"] is not None]),
        "control_success": _mean([r["solved1"] for r in rows if r["family"] == CONTROL]),
    }
    occ: dict[int, list] = defaultdict(list)
    for r in learnable:
        occ[r["occurrence"]].append(r["solved1"])
    m["success_by_occurrence"] = [_mean(occ[k]) for k in sorted(occ)]

    # Regressions: a rule satisfied on an earlier first attempt, violated on a later one, unchanged meanwhile.
    regressions = opportunities = 0
    satisfied: set[tuple[str, str]] = set()
    for r in learnable:
        for rule, ok in r["checks1"].items():
            if rule == BASE:
                continue
            key = (r["family"], rule)
            if key in satisfied:
                opportunities += 1
                regressions += not ok
            if ok:
                satisfied.add(key)
    m["regression_rate"] = regressions / opportunities if opportunities else None
    m["regressions"] = regressions

    # Drift: for each rule that replaced an older one, how many post-drift occurrences until it holds on a first try.
    drift = {}
    for rule in sorted({k for r in learnable for k in r["checks1"] if k.endswith(".v2")}):
        fam_rows = [r for r in learnable if rule in r["checks1"]]
        fam = fam_rows[0]["family"]
        old = rule[:-3] + ".v1"
        pre = [r for r in learnable if r["family"] == fam and old in r["checks1"]]
        hits = [r["checks1"][rule] for r in fam_rows]
        drift[rule] = {
            "knew_old_rule": bool(pre) and pre[-1]["checks1"][old],
            "post_drift_occurrences": len(hits),
            "stale_first_tries": hits.count(False),
            "repair_latency": hits.index(True) if True in hits else None,
        }
    m["drift"] = drift
    lat = [d["repair_latency"] for d in drift.values()]
    m["drift_unrepaired"] = sum(x is None for x in lat)
    m["drift_repair_latency"] = _mean([x for x in lat if x is not None])

    def tot(key):
        vals = [(r.get("usage1") or {}).get(key) for r in rows] + [(r.get("usage2") or {}).get(key) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) if vals else None

    m["input_tokens"], m["output_tokens"], m["cost_usd"] = tot("input_tokens"), tot("output_tokens"), tot("cost_usd")
    solved = sum(r["solved1"] for r in rows)
    m["cost_per_first_try_solve"] = m["cost_usd"] / solved if m["cost_usd"] is not None and solved else None
    m["library_final"] = rows[-1].get("library") if rows else None
    return m


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float] | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0], xs[0], xs[0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return statistics.fmean(xs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


HEADLINE = ["first_try_success", "post_first_success", "probe_success", "aulc", "feedback_uptake",
            "regression_rate", "drift_unrepaired", "drift_repair_latency", "control_success",
            "cost_per_first_try_solve"]


def collect(root: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """{(agent, condition): {seed: metrics}} from every records.jsonl under root."""
    out: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for p in sorted(root.rglob("records.jsonl")):
        rows = load(p)
        if rows:
            out[(rows[0]["agent"], rows[0]["condition"])][rows[0]["seed"]] = run_metrics(rows)
    return dict(out)


def summarize(root: Path) -> dict:
    runs = collect(root)
    table = {}
    for key, seeds in runs.items():
        table["/".join(key)] = {
            "seeds": sorted(seeds),
            **{h: bootstrap_ci([s[h] for s in seeds.values()]) for h in HEADLINE},
            "success_by_occurrence": [
                _mean([s["success_by_occurrence"][k] for s in seeds.values() if k < len(s["success_by_occurrence"])])
                for k in range(max(len(s["success_by_occurrence"]) for s in seeds.values()))
            ],
        }
    gains = {}
    for (agent, cond), seeds in runs.items():
        frozen = runs.get((agent, "frozen"))
        if cond != "learning" or not frozen:
            continue
        paired = [seeds[s]["post_first_success"] - frozen[s]["post_first_success"] for s in seeds if s in frozen]
        gains[agent] = bootstrap_ci(paired)
    return {"runs": table, "learning_gain_vs_frozen": gains}


def format_table(summary: dict) -> str:
    def f(ci):
        if ci is None:
            return "-"
        mean, lo, hi = ci
        return f"{mean:.2f}" if lo == hi else f"{mean:.2f} [{lo:.2f},{hi:.2f}]"

    cols = ["first_try_success", "post_first_success", "probe_success", "feedback_uptake",
            "regression_rate", "drift_unrepaired", "control_success"]
    lines = ["| run | seeds | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for name, row in sorted(summary["runs"].items()):
        lines.append(f"| {name} | {len(row['seeds'])} | " + " | ".join(f(row[c]) for c in cols) + " |")
    if summary["learning_gain_vs_frozen"]:
        lines += ["", "Learning gain (post-first-occurrence first-try success, learning minus frozen, paired by seed):"]
        for agent, ci in sorted(summary["learning_gain_vs_frozen"].items()):
            lines.append(f"- {agent}: {f(ci)}")
    return "\n".join(lines)
