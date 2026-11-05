"""Run one agent under one condition over one seeded episode stream and record every attempt."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .agents import Agent
from .families import BASE, FAMILIES, materialize
from .schedule import Episode, build

CONDITIONS = ("learning", "frozen")


def task_message(ep: Episode, ws: Path) -> str:
    return (f"This is a recurring task for our team. Work in {ws}.\n\n{ep.variant.prompt}\n"
            "Do the work directly; don't ask clarifying questions.")


def feedback_message(ep: Episode, checks: dict[str, bool]) -> str:
    fam = FAMILIES[ep.family]
    texts = {r.id: r.text for r in fam.rules(ep.epoch)}
    lines = [f"- {texts[k]}" for k, ok in checks.items() if not ok and k in texts]
    if not checks.get(BASE, True):
        lines.insert(0, f"- {fam.base_hint}")
    return ("Thanks, but this doesn't follow our team's conventions yet:\n" + "\n".join(lines)
            + "\nPlease fix the output in place. These conventions apply every time we do this task.")


def _score(checks: dict[str, bool]) -> float:
    return sum(checks.values()) / len(checks) if checks else 0.0


def run(agent: Agent, condition: str, seed: int, out_dir: Path, *, episodes: int = 40,
        drift_at: float = 0.6, retry: bool = True, hook=None) -> Path:
    """Writes records.jsonl under out_dir/<agent>/<condition>/seed-<n>/ and returns its path.

    hook(index, agent) runs between episodes; the Hermes adapter uses it to trigger the curator.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    run_dir = out_dir.resolve() / agent.name / condition / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    stream = build(seed, episodes, drift_at)
    agent.begin_run(run_dir)
    try:
        with records_path.open("w") as rec:
            for ep in stream:
                if condition == "frozen":
                    agent.reset_state()
                ws = run_dir / "ws" / f"ep{ep.index:03d}-{ep.family}"
                materialize(ep.variant, ws)
                agent.begin_episode(ep, ws)
                t0 = time.time()
                usage1 = agent.attempt(ep, ws, task_message(ep, ws), followup=False)
                checks1 = FAMILIES[ep.family].verify(ep.variant, ws)
                row = {
                    "agent": agent.name, "condition": condition, "seed": seed, "index": ep.index,
                    "family": ep.family, "occurrence": ep.occurrence, "epoch": ep.epoch, "probe": ep.probe,
                    "active_rules": FAMILIES[ep.family].rule_ids(ep.epoch),
                    "checks1": checks1, "score1": _score(checks1), "solved1": all(checks1.values()),
                    "usage1": usage1, "checks2": None, "solved2": None, "usage2": None,
                }
                if retry and not ep.probe and not row["solved1"]:
                    FAMILIES[ep.family].before_retry(ws)
                    failed = [k for k, ok in checks1.items() if not ok]
                    row["usage2"] = agent.attempt(ep, ws, feedback_message(ep, checks1), followup=True, violated=failed)
                    checks2 = FAMILIES[ep.family].verify(ep.variant, ws)
                    row.update(checks2=checks2, solved2=all(checks2.values()))
                row["seconds"] = round(time.time() - t0, 2)
                row["library"] = agent.end_episode(ep, ws)
                rec.write(json.dumps(row) + "\n")
                rec.flush()
                if hook:
                    hook(ep.index, agent)
    finally:
        agent.end_run()
    return records_path
