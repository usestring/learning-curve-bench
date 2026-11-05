"""Task families: generators, hidden house rules, verifiers and reference solvers.

Every family hides team conventions that an agent cannot infer from the task text. A rule
can be learned only from feedback on an earlier episode. Each family returns one check per
rule (plus a "base" check for doing the task at all), so a failure always says which rule
was missed. That is what makes regressions and stale knowledge measurable.

`solve(variant, known)` is the reference solver. Given the set of rule ids it knows, it
produces the output an agent with exactly that knowledge would write. With every active
rule known it must pass all checks (the tests assert this), and leaving out one rule must
fail exactly that rule's check.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

BASE = "base"


@dataclass(frozen=True)
class Rule:
    id: str
    slot: str  # rules in the same slot supersede each other across a drift
    text: str


@dataclass
class Variant:
    family: str
    epoch: int
    prompt: str
    files: dict[str, str]
    data: dict
    executables: tuple[str, ...] = ()


@dataclass
class Solution:
    files: dict[str, str] = field(default_factory=dict)
    commands: list[list[str]] = field(default_factory=list)  # argv relative to the workspace


class Family:
    name: str = ""
    base_hint: str = "The expected output file is missing or incomplete."

    def rules(self, epoch: int) -> list[Rule]:
        raise NotImplementedError

    def generate(self, rng: random.Random, epoch: int) -> Variant:
        raise NotImplementedError

    def solve(self, v: Variant, known: set[str]) -> Solution:
        raise NotImplementedError

    def verify(self, v: Variant, ws: Path) -> dict[str, bool]:
        raise NotImplementedError

    def before_retry(self, ws: Path) -> None:
        """Clear state a second attempt should not inherit."""

    def rule_ids(self, epoch: int) -> list[str]:
        return [r.id for r in self.rules(epoch)]


def _read(ws: Path, rel: str) -> str | None:
    p = ws / rel
    try:
        return p.read_text()
    except (OSError, UnicodeDecodeError):
        return None


WORDS_A = ["retry", "cache", "export", "login", "search", "billing", "webhook", "invoice", "sync", "upload"]
WORDS_B = ["timeout", "pagination", "rate limit", "dark mode", "CSV export", "audit log", "SSO", "filters"]


# --- A: release notes -------------------------------------------------------------------


class ReleaseNotes(Family):
    name = "release_notes"
    out = "out/RELEASE_NOTES.md"

    def rules(self, epoch):
        rs = [
            Rule("rn.header", "rn.header",
                 "Release notes start with a single line `# Release YYYY-MM-DD`, using the date of the most recent commit."),
            Rule("rn.order", "rn.order",
                 "Use exactly two sections, `## Fixes` first and then `## Features`, each commit under its own type."),
            Rule("rn.drop", "rn.drop", "Leave chore and docs commits out of the release notes entirely."),
        ]
        if epoch == 0:
            rs.append(Rule("rn.ticket.v1", "rn.ticket", "Format every entry as `- [ACME-123] message`."))
        else:
            rs.append(Rule("rn.ticket.v2", "rn.ticket",
                           "Ticket references now go at the end of the entry: `- message (ACME-123)`. "
                           "The `[ACME-123]` prefix format is retired."))
        return rs

    def generate(self, rng, epoch):
        types = ["feat", "fix", rng.choice(["chore", "docs"])]
        types += [rng.choice(["feat", "fix", "chore", "docs"]) for _ in range(rng.randint(2, 5))]
        rng.shuffle(types)
        commits, used = [], set()
        for t in types:
            while True:
                if t == "feat":
                    msg = f"Add {rng.choice(WORDS_B)} to {rng.choice(WORDS_A)}"
                elif t == "fix":
                    msg = f"Fix {rng.choice(WORDS_B)} in {rng.choice(WORDS_A)}"
                elif t == "chore":
                    msg = f"Bump {rng.choice(WORDS_A)} dependency"
                else:
                    msg = f"Document {rng.choice(WORDS_A)} {rng.choice(WORDS_B)}"
                if msg not in used:
                    used.add(msg)
                    break
            day = rng.randint(1, 28)
            commits.append({
                "sha": "%07x" % rng.getrandbits(28),
                "type": t,
                "ticket": f"ACME-{rng.randint(100, 999)}",
                "message": msg,
                "date": f"2026-10-{day:02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
            })
        prompt = ("commits.json lists the commits going into our next release. "
                  f"Write the release notes to {self.out}.")
        return Variant(self.name, epoch, prompt, {"commits.json": json.dumps(commits, indent=2) + "\n"},
                       {"commits": commits})

    @staticmethod
    def _latest(commits):
        return max(c["date"] for c in commits)[:10]

    def _entry(self, c, known):
        if "rn.ticket.v2" in known:
            return f"- {c['message']} ({c['ticket']})"
        if "rn.ticket.v1" in known:
            return f"- [{c['ticket']}] {c['message']}"
        return f"- {c['message']}"

    def solve(self, v, known):
        cs = v.data["commits"]
        feats = [c for c in cs if c["type"] == "feat"]
        fixes = [c for c in cs if c["type"] == "fix"]
        other = [c for c in cs if c["type"] in ("chore", "docs")]
        keep_other = "rn.drop" not in known
        lines = [f"# Release {self._latest(cs)}" if "rn.header" in known else "# Release notes", ""]
        if "rn.order" in known:
            sections = [("## Fixes", fixes), ("## Features", feats + (other if keep_other else []))]
        else:
            sections = [("## Features", feats), ("## Fixes", fixes)]
            if keep_other:
                sections.append(("## Other", other))
        for head, items in sections:
            lines += [head, *[self._entry(c, known) for c in items], ""]
        return Solution(files={self.out: "\n".join(lines)})

    def verify(self, v, ws):
        text = _read(ws, self.out)
        ids = self.rule_ids(v.epoch)
        if text is None:
            return {BASE: False, **{r: False for r in ids}}
        cs = v.data["commits"]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kept = [c for c in cs if c["type"] in ("feat", "fix")]
        sections: dict[str, list[str]] = {}
        heads: list[str] = []
        cur = None
        for ln in lines[1:]:
            if ln.startswith("## "):
                cur = ln
                heads.append(ln)
                sections[cur] = []
            elif cur is not None:
                sections[cur].append(ln)
        checks = {BASE: all(c["message"] in text for c in kept)}
        checks["rn.header"] = bool(lines) and lines[0] == f"# Release {self._latest(cs)}"
        checks["rn.order"] = heads == ["## Fixes", "## Features"] and all(
            any(c["message"] in ln for ln in sections.get("## Fixes" if c["type"] == "fix" else "## Features", []))
            for c in kept)
        checks["rn.drop"] = not any(c["message"] in text for c in cs if c["type"] in ("chore", "docs"))
        tid = ids[-1]
        checks[tid] = all(self._entry(c, {tid}) in lines for c in kept)
        return checks


# --- B: internal CLI with silent conventions --------------------------------------------

ACMECTL = r'''#!/usr/bin/env python3
"""acmectl - Acme internal deploy tool. Ask in #platform for usage."""
import json, os, sys, time
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
state = os.path.join(root, ".acmectl")
os.makedirs(state, exist_ok=True)
args = sys.argv[1:]
if not args or args[0] in ("-h", "--help", "help"):
    print("usage: acmectl <command> [args...]\ncommands: deploy, lock, unlock, status")
    sys.exit(0)
try:
    major = open(os.path.join(state, "version")).read().strip()
except OSError:
    major = "1"
with open(os.path.join(state, "log.jsonl"), "a") as f:
    f.write(json.dumps({"argv": args, "ts": time.time()}) + "\n")
if args[0] == "deploy" and "--env" in args and major == "2":
    print("warning: --env is deprecated and ignored in acmectl 2.x", file=sys.stderr)
print("ok")
'''


class Acmectl(Family):
    name = "acmectl"
    services = ["billing-api", "search-svc", "auth-gw", "reports-worker", "notify-svc"]

    def rules(self, epoch):
        rs = [
            Rule("ac.lock", "ac.lock",
                 "Always run `acmectl lock <service>` before deploying and `acmectl unlock <service>` afterwards."),
            Rule("ac.vprefix", "ac.vprefix", "Pass versions to acmectl with a leading v, e.g. `v2.4.1`."),
        ]
        if epoch == 0:
            rs.append(Rule("ac.env.v1", "ac.env", "Choose the environment with `--env <name>`."))
        else:
            rs.append(Rule("ac.env.v2", "ac.env",
                           "acmectl 2.x replaced `--env` with `--target <name>`; `--env` is now ignored."))
        return rs

    def generate(self, rng, epoch):
        svc = rng.choice(self.services)
        ver = f"{rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 9)}"
        prompt = f"Deploy {svc} version {ver} to staging using our acmectl tool in bin/."
        files = {"bin/acmectl": ACMECTL, ".acmectl/version": f"{epoch + 1}\n"}
        return Variant(self.name, epoch, prompt, files, {"service": svc, "version": ver},
                       executables=("bin/acmectl",))

    def solve(self, v, known):
        svc, ver = v.data["service"], v.data["version"]
        version = f"v{ver}" if "ac.vprefix" in known else ver
        if "ac.env.v2" in known:
            env = ["--target", "staging"]
        elif "ac.env.v1" in known:
            env = ["--env", "staging"]
        else:
            env = ["--environment", "staging"]
        deploy = ["bin/acmectl", "deploy", svc, version, *env]
        if "ac.lock" in known:
            return Solution(commands=[["bin/acmectl", "lock", svc], deploy, ["bin/acmectl", "unlock", svc]])
        return Solution(commands=[deploy])

    @staticmethod
    def _calls(ws):
        text = _read(ws, ".acmectl/log.jsonl") or ""
        calls = []
        for ln in text.splitlines():
            try:
                calls.append(json.loads(ln)["argv"])
            except (ValueError, KeyError, TypeError):
                pass
        return calls

    def verify(self, v, ws):
        svc, ver = v.data["service"], v.data["version"]
        calls = self._calls(ws)
        ids = self.rule_ids(v.epoch)
        deploys = [i for i, a in enumerate(calls) if a[:1] == ["deploy"] and svc in a]
        if not deploys:
            return {BASE: False, **{r: False for r in ids}}
        last = deploys[-1]
        d = calls[last]
        checks = {BASE: True}
        checks["ac.lock"] = (["lock", svc] in calls[:last]) and (["unlock", svc] in calls[last + 1:])
        checks["ac.vprefix"] = f"v{ver}" in d and ver not in d
        if v.epoch == 0:
            checks["ac.env.v1"] = _flag(d, "--env") == "staging"
        else:
            checks["ac.env.v2"] = _flag(d, "--target") == "staging" and "--env" not in d
        return checks

    def before_retry(self, ws):
        log = ws / ".acmectl/log.jsonl"
        if log.exists():
            log.rename(ws / ".acmectl/log.attempt1.jsonl")


def _flag(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# --- C: CSV cleanup -----------------------------------------------------------------------

FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea"]
LAST = ["Ruiz", "Okafor", "Li", "Nair", "Stone", "Berg", "Moreau", "Sato", "Kowal", "Diaz"]


class CsvCleanup(Family):
    name = "csv_cleanup"
    out = "out/clean.csv"

    def rules(self, epoch):
        return [
            Rule("cs.cents", "cs.cents", "Amounts are stored as integer cents in a column named `amount_cents`."),
            Rule("cs.dedupe", "cs.dedupe",
                 "Emails are lowercased, and rows with a duplicate email are dropped, keeping the first occurrence."),
            Rule("cs.dates", "cs.dates", "Raw exports use DD/MM/YYYY dates; write them as ISO `YYYY-MM-DD`."),
            Rule("cs.columns", "cs.columns",
                 "Column order is `email,name,amount_cents,date`, with rows sorted by date ascending "
                 "(ties keep input order)."),
        ]

    def generate(self, rng, epoch):
        people = rng.sample([(f, l) for f in FIRST for l in LAST], rng.randint(5, 8))
        rows = []
        for f, l in people:
            d, m = rng.randint(1, 28), rng.randint(1, 12)
            email = f"{f}.{l}@Example.com" if rng.random() < 0.5 else f"{f.lower()}.{l.lower()}@example.com"
            rows.append({"name": f"{f} {l}", "email": email, "amount": f"{rng.randint(1, 900)}.{rng.randint(0, 99):02d}",
                         "d": d, "m": m})
        for src in rng.sample(rows, 2):
            # Duplicates are dated no earlier than the original, so sorting never reorders first occurrences.
            dup = dict(src, email=src["email"].upper(), amount=f"{rng.randint(1, 900)}.{rng.randint(0, 99):02d}",
                       d=min(28, src["d"] + rng.randint(0, 5)))
            rows.insert(rng.randint(rows.index(src) + 1, len(rows)), dup)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["name", "email", "amount", "date"])
        for r in rows:
            w.writerow([r["name"], r["email"], r["amount"], f"{r['d']:02d}/{r['m']:02d}/2026"])
        prompt = f"Clean up raw.csv (a customer payments export) and write the result to {self.out}."
        return Variant(self.name, epoch, prompt, {"raw.csv": buf.getvalue()}, {"rows": rows})

    @staticmethod
    def _iso(r):
        return f"2026-{r['m']:02d}-{r['d']:02d}"

    @staticmethod
    def _cents(r):
        whole, frac = r["amount"].split(".")
        return int(whole) * 100 + int(frac)

    def _expected(self, rows):
        seen, out = set(), []
        for r in rows:
            if r["email"].lower() not in seen:
                seen.add(r["email"].lower())
                out.append(r)
        return sorted(out, key=lambda r: (r["m"], r["d"]))

    def solve(self, v, known):
        rows = list(v.data["rows"])
        if "cs.dedupe" in known:
            seen, kept = set(), []
            for r in rows:
                if r["email"].lower() not in seen:
                    seen.add(r["email"].lower())
                    kept.append(r)
            rows = kept
        if "cs.columns" in known:
            rows = sorted(rows, key=lambda r: (r["m"], r["d"]))
        amount_col = "amount_cents" if "cs.cents" in known else "amount"
        cols = ["email", "name", amount_col, "date"] if "cs.columns" in known else ["name", "email", amount_col, "date"]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            vals = {
                "name": r["name"],
                "email": r["email"].lower() if "cs.dedupe" in known else r["email"],
                amount_col: str(self._cents(r)) if "cs.cents" in known else r["amount"],
                "date": self._iso(r) if "cs.dates" in known else f"{r['d']:02d}/{r['m']:02d}/2026",
            }
            w.writerow([vals[c] for c in cols])
        return Solution(files={self.out: buf.getvalue()})

    def verify(self, v, ws):
        text = _read(ws, self.out)
        ids = self.rule_ids(v.epoch)
        fail = {BASE: False, **{r: False for r in ids}}
        if not text:
            return fail
        try:
            table = list(csv.reader(io.StringIO(text)))
        except csv.Error:
            return fail
        if len(table) < 2:
            return fail
        header = [h.strip() for h in table[0]]
        body = [dict(zip(header, [c.strip() for c in row])) for row in table[1:] if row]
        raw = v.data["rows"]
        by_email: dict[str, list[dict]] = {}
        for r in raw:
            by_email.setdefault(r["email"].lower(), []).append(r)
        amt = next((h for h in header if h.startswith("amount")), None)
        if "email" not in header or "date" not in header or amt is None:
            return fail
        emails = [b.get("email", "") for b in body]
        expected = self._expected(raw)
        checks = {BASE: all(e.lower() in by_email for e in emails)}
        checks["cs.cents"] = amt == "amount_cents" and all(
            b[amt].isdigit() and int(b[amt]) in {self._cents(r) for r in by_email.get(b["email"].lower(), [])}
            for b in body)
        checks["cs.dedupe"] = (all(e == e.lower() for e in emails) and len(set(emails)) == len(emails)
                               and set(emails) == {r["email"].lower() for r in expected})
        checks["cs.dates"] = all(
            b["date"] in {self._iso(r) for r in by_email.get(b["email"].lower(), [])} for b in body)
        firsts = list(dict.fromkeys(e.lower() for e in emails))
        checks["cs.columns"] = (len(header) == 4 and header[0] == "email" and header[1] == "name"
                                and header[2] == amt and header[3] == "date"
                                and firsts == [r["email"].lower() for r in expected])
        return checks


# --- D: incident triage policy ------------------------------------------------------------


class Triage(Family):
    name = "triage"
    out = "out/triage.json"

    def rules(self, epoch):
        return [
            Rule("tr.impact", "tr.impact", "Any customer-impacting incident is at least SEV2."),
            Rule("tr.payments", "tr.payments", "Anything touching the payments service is at least SEV2."),
            Rule("tr.eu", "tr.eu", "Customer-impacting incidents in the eu region are SEV1 (data-residency obligations)."),
        ]

    @staticmethod
    def _binding(i):
        if i["outage"]:
            return BASE
        if i["customer_impact"] and i["region"] == "eu":
            return "tr.eu"
        if i["customer_impact"]:
            return "tr.impact"
        if i["service"] == "payments":
            return "tr.payments"
        return BASE

    @staticmethod
    def _sev(i, known):
        if i["outage"]:
            return "SEV1"
        if "tr.eu" in known and i["customer_impact"] and i["region"] == "eu":
            return "SEV1"
        if "tr.impact" in known and i["customer_impact"]:
            return "SEV2"
        if "tr.payments" in known and i["service"] == "payments":
            return "SEV2"
        return "SEV3"

    def generate(self, rng, epoch):
        nonpay = ["search", "auth", "reports", "notifications"]
        regions = ["us", "eu", "apac"]
        seeds = [
            dict(service=rng.choice(nonpay + ["payments"]), region=rng.choice(regions), customer_impact=rng.random() < .5, outage=True),
            dict(service=rng.choice(nonpay), region="eu", customer_impact=True, outage=False),
            dict(service=rng.choice(nonpay), region=rng.choice(["us", "apac"]), customer_impact=True, outage=False),
            dict(service="payments", region=rng.choice(regions), customer_impact=False, outage=False),
            dict(service=rng.choice(nonpay), region=rng.choice(regions), customer_impact=False, outage=False),
        ]
        for _ in range(rng.randint(1, 4)):
            seeds.append(dict(service=rng.choice(nonpay + ["payments"]), region=rng.choice(regions),
                              customer_impact=rng.random() < .5, outage=rng.random() < .15))
        rng.shuffle(seeds)
        titles = ["Elevated 5xx", "Slow queries", "Stuck queue", "Cert expiry warning", "Login failures", "Data lag"]
        incidents = [dict(id=f"INC-{rng.randint(1000, 9999)}-{n}", title=rng.choice(titles), **s) for n, s in enumerate(seeds)]
        prompt = ("Assign a severity (SEV1, SEV2 or SEV3) to each incident in incidents.json. "
                  f"Write {self.out} as a JSON object mapping incident id to severity.")
        return Variant(self.name, epoch, prompt, {"incidents.json": json.dumps(incidents, indent=2) + "\n"},
                       {"incidents": incidents})

    def solve(self, v, known):
        out = {i["id"]: self._sev(i, known) for i in v.data["incidents"]}
        return Solution(files={self.out: json.dumps(out, indent=2) + "\n"})

    def verify(self, v, ws):
        ids = self.rule_ids(v.epoch)
        text = _read(ws, self.out)
        try:
            got = json.loads(text) if text else None
        except ValueError:
            got = None
        incs = v.data["incidents"]
        if not isinstance(got, dict) or any(str(got.get(i["id"])).upper() not in ("SEV1", "SEV2", "SEV3") for i in incs):
            return {BASE: False, **{r: False for r in ids}}
        allk = set(ids)
        checks = {}
        for key in [BASE, *ids]:
            checks[key] = all(str(got[i["id"]]).upper() == self._sev(i, allk) for i in incs if self._binding(i) == key)
        return checks


# --- control: no hidden rules ---------------------------------------------------------------


class LogCount(Family):
    """A task with no team conventions. A drop here means learned behaviour is interfering."""

    name = "log_count"
    out = "out/answer.txt"

    def rules(self, epoch):
        return []

    def generate(self, rng, epoch):
        levels = ["INFO", "DEBUG", "WARN", "ERROR"]
        lines, n = [], 0
        for k in range(rng.randint(30, 80)):
            lv = rng.choices(levels, weights=[5, 3, 2, 1])[0]
            n += lv == "ERROR"
            noise = " (previous ERROR cleared)" if lv != "ERROR" and rng.random() < .1 else ""
            lines.append(f"2026-10-{rng.randint(1, 28):02d}T10:{k % 60:02d}:00Z {lv} worker-{rng.randint(1, 9)} job done{noise}")
        prompt = f"How many lines in app.log are at ERROR level? Write just the number to {self.out}."
        return Variant(self.name, epoch, prompt, {"app.log": "\n".join(lines) + "\n"}, {"count": n})

    def solve(self, v, known):
        return Solution(files={self.out: f"{v.data['count']}\n"})

    def verify(self, v, ws):
        return {BASE: (_read(ws, self.out) or "").strip() == str(v.data["count"])}


FAMILIES: dict[str, Family] = {f.name: f for f in (ReleaseNotes(), Acmectl(), CsvCleanup(), Triage(), LogCount())}


def materialize(v: Variant, ws: Path) -> None:
    if ws.exists():
        shutil.rmtree(ws)
    for rel, content in v.files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for rel in v.executables:
        p = ws / rel
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (ws / "out").mkdir(exist_ok=True)


def apply_solution(sol: Solution, ws: Path) -> None:
    """Write a reference solution into a workspace the way an agent would."""
    import subprocess
    import sys

    for rel, content in sol.files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for argv in sol.commands:
        subprocess.run([sys.executable, str(ws / argv[0]), *argv[1:]], cwd=ws, check=True,
                       capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
