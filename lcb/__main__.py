"""CLI: python3 -m lcb {run,report,validate,show}."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import metrics
from .agents import REFERENCE_AGENTS
from .runner import CONDITIONS, feedback_message, run, task_message
from .schedule import build


def _make_agent(args):
    if args.agent in REFERENCE_AGENTS:
        return REFERENCE_AGENTS[args.agent](), None
    if args.agent == "hermes":
        from .hermes import HermesAgent

        if not args.hermes_home_template:
            sys.exit("--hermes-home-template is required for --agent hermes")
        agent = HermesAgent(Path(args.hermes_home_template), mode=args.hermes_mode, hermes_bin=args.hermes_bin,
                            timeout=args.timeout, review_grace=args.review_grace, max_turns=args.max_turns,
                            toolsets=args.toolsets, label=args.label)
        hook = None
        if args.curator_every:
            def hook(i, a):
                if (i + 1) % args.curator_every == 0:
                    a.run_curator(consolidate=args.curator_consolidate)
        return agent, hook
    sys.exit(f"unknown agent {args.agent}")


def cmd_run(args):
    for seed in args.seeds:
        agent, hook = _make_agent(args)
        p = run(agent, args.condition, seed, Path(args.out), episodes=args.episodes, drift_at=args.drift_at,
                retry=not args.no_retry, hook=hook)
        print(f"{agent.name}/{args.condition}/seed-{seed}: {p}")


def cmd_report(args):
    s = metrics.summarize(Path(args.root))
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print(metrics.format_table(s))


def validate(out: Path, seeds=(0, 1, 2), episodes: int = 40) -> dict:
    """Run every reference agent under both conditions; return the summary."""
    for name, make in REFERENCE_AGENTS.items():
        for cond in CONDITIONS:
            for seed in seeds:
                run(make(), cond, seed, out, episodes=episodes)
    return metrics.summarize(out)


def cmd_validate(args):
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="lcb-validate-"))
    s = validate(out, tuple(args.seeds), args.episodes)
    print(metrics.format_table(s))
    print(f"\nruns written to {out}")


def cmd_show(args):
    for ep in build(args.seed, args.episodes, args.drift_at):
        if args.index is not None and ep.index != args.index:
            continue
        tag = " probe" if ep.probe else ""
        print(f"[{ep.index:02d}] {ep.family} occ={ep.occurrence} epoch={ep.epoch}{tag}")
        if args.index is not None:
            ws = Path("<workspace>")
            print("\n" + task_message(ep, ws))
            from .families import FAMILIES
            print("\nFeedback if every rule were missed:\n" + feedback_message(
                ep, {k: False for k in ["base", *FAMILIES[ep.family].rule_ids(ep.epoch)]}))
            for rel, content in ep.variant.files.items():
                print(f"\n--- {rel} ---\n{content[:1500]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lcb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one agent under one condition")
    r.add_argument("--agent", required=True, help=f"{', '.join(REFERENCE_AGENTS)} or hermes")
    r.add_argument("--condition", choices=CONDITIONS, default="learning")
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--episodes", type=int, default=40)
    r.add_argument("--drift-at", type=float, default=0.6)
    r.add_argument("--no-retry", action="store_true", help="skip the feedback retry (feedback is still never given on probes)")
    r.add_argument("--out", default="runs")
    h = r.add_argument_group("hermes")
    h.add_argument("--hermes-home-template", help="dir with config.yaml/.env copied into a fresh HERMES_HOME")
    h.add_argument("--hermes-mode", choices=["acp", "oneshot"], default="acp")
    h.add_argument("--hermes-bin", default="hermes")
    h.add_argument("--label", help="run name, e.g. hermes-acp-context-only")
    h.add_argument("--toolsets", help="oneshot only: -t value, e.g. terminal,file,memory,session_search")
    h.add_argument("--timeout", type=float, default=900)
    h.add_argument("--review-grace", type=float, default=45, help="seconds to wait after each episode (acp)")
    h.add_argument("--max-turns", type=int, default=40)
    h.add_argument("--curator-every", type=int, default=0, help="run `hermes curator run` every N episodes")
    h.add_argument("--curator-consolidate", action="store_true")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="summarize every records.jsonl under a directory")
    p.add_argument("root")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)

    v = sub.add_parser("validate", help="run the reference agents (offline, free) and print the table")
    v.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    v.add_argument("--episodes", type=int, default=40)
    v.add_argument("--out")
    v.set_defaults(fn=cmd_validate)

    s = sub.add_parser("show", help="print the episode stream, or one episode's prompt and files")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--episodes", type=int, default=40)
    s.add_argument("--drift-at", type=float, default=0.6)
    s.add_argument("--index", type=int)
    s.set_defaults(fn=cmd_show)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
