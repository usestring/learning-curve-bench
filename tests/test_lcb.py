import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcb import metrics  # noqa: E402
from lcb.__main__ import validate  # noqa: E402
from lcb.families import BASE, FAMILIES, apply_solution, materialize  # noqa: E402
from lcb.hermes import HermesAgent, _harvest_usage  # noqa: E402
from lcb.runner import run  # noqa: E402
from lcb.schedule import build  # noqa: E402

SEEDS = range(25)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def checks_for(self, fam, v, known, tag):
        ws = self.tmp / tag
        materialize(v, ws)
        apply_solution(fam.solve(v, known), ws)
        return fam.verify(v, ws)


class VerifierTests(Tmp):
    def test_full_knowledge_passes_every_check(self):
        for fam in FAMILIES.values():
            for epoch in (0, 1):
                for s in SEEDS:
                    v = fam.generate(random.Random(s), epoch)
                    checks = self.checks_for(fam, v, set(fam.rule_ids(epoch)), f"{fam.name}-{epoch}-{s}")
                    self.assertTrue(all(checks.values()), (fam.name, epoch, s, checks))

    def test_each_missing_rule_fails_only_its_own_check(self):
        for fam in FAMILIES.values():
            for epoch in (0, 1):
                ids = fam.rule_ids(epoch)
                for rule in ids:
                    for s in range(8):
                        v = fam.generate(random.Random(s), epoch)
                        checks = self.checks_for(fam, v, set(ids) - {rule}, f"{fam.name}-{rule}-{s}")
                        self.assertFalse(checks[rule], (fam.name, rule, s))
                        others = {k: ok for k, ok in checks.items() if k != rule}
                        self.assertTrue(all(others.values()), (fam.name, rule, s, checks))

    def test_no_knowledge_fails_every_rule_but_does_the_task(self):
        for fam in FAMILIES.values():
            for epoch in (0, 1):
                for s in range(8):
                    v = fam.generate(random.Random(s), epoch)
                    checks = self.checks_for(fam, v, set(), f"{fam.name}-none-{epoch}-{s}")
                    self.assertTrue(checks[BASE], (fam.name, s))
                    self.assertFalse(any(ok for k, ok in checks.items() if k != BASE), (fam.name, s, checks))

    def test_stale_rule_fails_after_drift(self):
        for name, old in (("release_notes", "rn.ticket.v1"), ("acmectl", "ac.env.v1")):
            fam = FAMILIES[name]
            v = fam.generate(random.Random(3), 1)
            known = (set(fam.rule_ids(1)) - {old.replace("v1", "v2")}) | {old}
            checks = self.checks_for(fam, v, known, f"stale-{name}")
            self.assertFalse(checks[old.replace("v1", "v2")])

    def test_missing_output_fails_base(self):
        for fam in FAMILIES.values():
            v = fam.generate(random.Random(0), 0)
            ws = self.tmp / f"empty-{fam.name}"
            materialize(v, ws)
            self.assertFalse(fam.verify(v, ws)[BASE], fam.name)

    def test_acmectl_warns_on_env_after_drift(self):
        fam = FAMILIES["acmectl"]
        v = fam.generate(random.Random(1), 1)
        ws = self.tmp / "acme"
        materialize(v, ws)
        p = subprocess.run([str(ws / "bin/acmectl"), "deploy", v.data["service"], "v1", "--env", "staging"],
                           cwd=ws, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("deprecated", p.stderr)


class ScheduleTests(unittest.TestCase):
    def test_deterministic_and_well_formed(self):
        a, b = build(7), build(7)
        self.assertEqual([(e.family, e.variant.files) for e in a], [(e.family, e.variant.files) for e in b])
        self.assertTrue(all(not e.probe for e in a if e.occurrence == 0))
        self.assertTrue(any(e.probe for e in a))
        self.assertEqual({e.epoch for e in a[:24]}, {0})
        self.assertEqual({e.epoch for e in a[24:]}, {1})
        counts = {f: sum(e.family == f for e in a) for f in FAMILIES}
        self.assertEqual(set(counts.values()), {8})


class MetricTests(unittest.TestCase):
    """The reference agents encode known behaviour; the metrics must recover it."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.s = validate(Path(cls._td.name), seeds=(0, 1), episodes=40)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def m(self, run, key):
        return self.s["runs"][run][key][0]

    def test_floor_and_ceiling(self):
        self.assertEqual(self.m("oracle/learning", "first_try_success"), 1.0)
        self.assertEqual(self.m("null/learning", "first_try_success"), 0.0)
        self.assertEqual(self.s["learning_gain_vs_frozen"]["null"][0], 0.0)
        self.assertEqual(self.s["learning_gain_vs_frozen"]["oracle"][0], 0.0)

    def test_learning_gain(self):
        self.assertGreater(self.s["learning_gain_vs_frozen"]["notebook"][0], 0.7)
        self.assertEqual(self.m("notebook/frozen", "post_first_success"), 0.0)

    def test_staleness(self):
        self.assertEqual(self.m("notebook/learning", "drift_unrepaired"), 0)
        self.assertEqual(self.m("stale-notebook/learning", "drift_unrepaired"), 2)
        self.assertLess(self.m("stale-notebook/learning", "post_first_success"),
                        self.m("notebook/learning", "post_first_success"))

    def test_regressions(self):
        self.assertEqual(self.m("notebook/learning", "regression_rate"), 0.0)
        self.assertGreater(self.m("forgetful-notebook-4/learning", "regression_rate"), 0.3)

    def test_control_unaffected(self):
        for run_name in self.s["runs"]:
            self.assertEqual(self.m(run_name, "control_success"), 1.0, run_name)


FAKE_ACP = textwrap.dedent('''\
    import json, os, sys
    def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
    cwd = {}
    for line in sys.stdin:
        m = json.loads(line)
        if m.get("method") == "initialize":
            send({"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": 1}})
        elif m.get("method") == "session/new":
            sid = "s%d" % len(cwd); cwd[sid] = m["params"]["cwd"]
            send({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": sid}})
        elif m.get("method") == "session/prompt":
            sid = m["params"]["sessionId"]
            send({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission", "params": {
                "sessionId": sid, "options": [{"optionId": "deny", "kind": "reject_once"},
                                              {"optionId": "allow_once", "kind": "allow_once"}]}})
            reply = json.loads(sys.stdin.readline())
            text = m["params"]["prompt"][0]["text"]
            with open(os.path.join(cwd[sid], "out", "seen.txt"), "a") as f:
                f.write(reply["result"]["outcome"]["optionId"] + "|" + text[:40] + "\\n")
            send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid,
                  "update": {"sessionUpdate": "usage_update", "usage": {"input_tokens": 100, "output_tokens": 7}}}})
            send({"jsonrpc": "2.0", "id": m["id"], "result": {"stopReason": "end_turn"}})
''')


class HermesAdapterTests(Tmp):
    def _bin(self, body):
        b = self.tmp / "fake-hermes"
        b.write_text(f"#!{sys.executable}\n" + body)
        b.chmod(0o755)
        tpl = self.tmp / "tpl"
        tpl.mkdir()
        (tpl / "config.yaml").write_text("model: fake\n")
        return b, tpl

    def test_acp_protocol_session_per_episode_and_permission(self):
        b, tpl = self._bin(FAKE_ACP)
        agent = HermesAgent(tpl, mode="acp", hermes_bin=str(b), review_grace=0, timeout=30)
        path = run(agent, "learning", 0, self.tmp / "runs", episodes=3)
        rows = metrics.load(path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["usage1"]["input_tokens"], 100)
        self.assertEqual(rows[0]["usage1"]["stop_reason"], "end_turn")
        seen = list((path.parent / "ws").glob("*/out/seen.txt"))
        self.assertEqual(len(seen), 3)
        first = seen[0].read_text().splitlines()
        self.assertTrue(first[0].startswith("allow_once|This is a recurring task"))
        self.assertTrue(first[1].startswith("allow_once|Thanks, but"))  # retry lands in the same session
        self.assertEqual(len(list((path.parent / "homes").iterdir())), 1)

    def test_relative_output_preserves_acp_workspace_paths(self):
        b, tpl = self._bin(FAKE_ACP)
        relative_out = Path(os.path.relpath(self.tmp / "relative-runs"))
        path = run(HermesAgent(tpl, hermes_bin=str(b), review_grace=0, timeout=30),
                   "learning", 0, relative_out, episodes=3)
        self.assertTrue(path.is_absolute())
        self.assertEqual(len(metrics.load(path)), 3)
        self.assertEqual(len(list((path.parent / "ws").glob("*/out/seen.txt"))), 3)
        self.assertTrue((path.parent / "homes/home-001/config.yaml").exists())

    def test_frozen_gets_fresh_home_per_episode(self):
        b, tpl = self._bin(FAKE_ACP)
        path = run(HermesAgent(tpl, hermes_bin=str(b), review_grace=0, timeout=30), "frozen", 0,
                   self.tmp / "runs", episodes=3)
        self.assertEqual(len(list((path.parent / "homes").iterdir())), 4)  # one at start + one per episode
        self.assertTrue((path.parent / "homes" / "home-004" / "config.yaml").exists())

    def test_oneshot_resumes_session_on_retry(self):
        b, tpl = self._bin(textwrap.dedent('''\
            import json, os, sys
            with open(os.path.join(os.environ["HERMES_HOME"], "argv.log"), "a") as f:
                f.write(json.dumps(sys.argv[1:]) + "\\n")
            print(json.dumps({"type": "system", "subtype": "init", "session_id": "abc"}))
            print(json.dumps({"type": "result", "session_id": "abc", "exit_code": 0,
                              "tokens": {"input": 5, "output": 2, "total": 7,
                                         "cache_read": 0, "cache_write": 0}}))
        '''))
        path = run(HermesAgent(tpl, mode="oneshot", hermes_bin=str(b), timeout=30), "learning", 0,
                   self.tmp / "runs", episodes=1)
        row = metrics.load(path)[0]
        self.assertEqual(row["usage1"]["input_tokens"], 5)
        self.assertEqual(row["usage1"]["output_tokens"], 2)
        calls = [json.loads(x) for x in (path.parent / "homes/home-001/argv.log").read_text().splitlines()]
        self.assertEqual(calls[0][:2], ["chat", "-q"])
        self.assertNotIn("-r", calls[0])
        if row["usage2"] is not None:
            self.assertEqual(calls[1][calls[1].index("-r") + 1], "abc")

    def test_usage_harvest_reads_stream_json_tokens(self):
        acc = {}
        _harvest_usage({"type": "result", "session_id": "abc", "exit_code": 0,
                        "tokens": {"input": 120, "output": 30, "total": 150,
                                   "cache_read": 10, "cache_write": 0}}, acc)
        self.assertEqual(acc, {"input_tokens": 120, "output_tokens": 30})

    def test_usage_harvest_handles_nested_and_alternate_names(self):
        acc = {}
        _harvest_usage([{"a": {"usage": {"inputTokens": 3, "outputTokens": 1}}}, {"usage": {"input_tokens": 2}}], acc)
        self.assertEqual(acc, {"input_tokens": 5, "output_tokens": 1})
        _harvest_usage({"usage": {"input_tokens": 4, "cost": 0.01},
                        "tokens": {"input": 999, "output": 999}}, acc)
        self.assertEqual(acc, {"input_tokens": 9, "output_tokens": 1, "cost_usd": 0.01})


if __name__ == "__main__":
    unittest.main()
