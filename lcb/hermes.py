"""Hermes Agent adapter (NousResearch/hermes-agent). Not yet run against a live model.

Two modes:

- acp (default): one long-lived `hermes acp` process per run, speaking the Agent Client
  Protocol (JSON-RPC 2.0, one JSON object per line on stdio). Each episode is a new ACP
  session, and the retry is a second prompt in the same session. Because the process stays
  up, Hermes's after-turn background review (agent/background_review.py) gets to run and
  write skills/memory, which is the self-improvement loop being measured.
- oneshot: `hermes chat -q ... --format stream-json` per attempt, with `-r <session>` for the
  retry. Hermes disables background review in finite one-shot turns
  (`load_background_review_settings()` returns disabled when `finite_turn_required()`), so
  this mode measures only what the agent saves with its own in-turn tool calls. Run it next
  to acp to show how much the harness mode changes the result.

State isolation uses HERMES_HOME. `home_template` is a directory holding the user's
config.yaml and .env (model, provider, API key). It is copied, never modified, and never
belongs in git. learning: one copy for the whole run. frozen: a fresh copy per episode.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .agents import Agent
from .schedule import Episode


def _num(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return v
    return None


def _harvest_usage(obj, acc: dict) -> None:
    """Sum token and cost fields wherever they appear in an event. Field names vary by version."""
    if isinstance(obj, dict):
        u = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
        if u:
            for out, keys in (("input_tokens", ("input_tokens", "prompt_tokens", "inputTokens")),
                              ("output_tokens", ("output_tokens", "completion_tokens", "outputTokens")),
                              ("cost_usd", ("cost_usd", "cost", "total_cost"))):
                v = _num(u, *keys)
                if v is not None:
                    acc[out] = acc.get(out, 0) + v
        for v in obj.values():
            if isinstance(v, (dict, list)) and v is not u:
                _harvest_usage(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _harvest_usage(v, acc)


class AcpClient:
    def __init__(self, argv: list[str], env: dict, cwd: Path, log: Path):
        self.stderr = open(log.with_suffix(".stderr"), "a")
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, env=env, cwd=cwd, text=True, bufsize=1)
        self.log = open(log, "a")
        self.next_id = 0
        self.pending: dict[int, queue.Queue] = {}
        self.updates: list[dict] = []
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()  # the reader thread answers permission requests concurrently
        threading.Thread(target=self._read, daemon=True).start()

    def _send(self, msg: dict) -> None:
        line = json.dumps(msg)
        with self.write_lock:
            self.log.write(">> " + line + "\n")
            self.log.flush()
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.log.write("<< " + line)
            self.log.flush()
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
                q = self.pending.pop(msg["id"], None)
                if q:
                    q.put(msg)
            elif msg.get("method") == "session/request_permission":
                opts = (msg.get("params") or {}).get("options") or []
                pick = next((o["optionId"] for o in opts if o.get("kind") == "allow_once"),
                            opts[0]["optionId"] if opts else "allow_once")
                self._send({"jsonrpc": "2.0", "id": msg["id"],
                            "result": {"outcome": {"outcome": "selected", "optionId": pick}}})
            elif "id" in msg and "method" in msg:
                # The client advertises no fs/terminal capabilities, so any other request is unexpected.
                self._send({"jsonrpc": "2.0", "id": msg["id"],
                            "error": {"code": -32601, "message": "method not supported by lcb client"}})
            elif msg.get("method") == "session/update":
                with self.lock:
                    self.updates.append(msg.get("params") or {})
        # EOF: the process exited, so fail waiting requests now instead of at their timeout.
        for q in list(self.pending.values()):
            q.put({"error": {"message": f"hermes acp exited (code {self.proc.poll()})"}})
        self.pending.clear()

    def request(self, method: str, params: dict, timeout: float) -> dict:
        self.next_id += 1
        q: queue.Queue = queue.Queue()
        self.pending[self.next_id] = q
        self._send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{method} timed out after {timeout}s") from None
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg["result"]

    def take_updates(self) -> list[dict]:
        with self.lock:
            out, self.updates = self.updates, []
        return out

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.proc.stdout.close()
        self.stderr.close()
        self.log.close()


class HermesAgent(Agent):
    def __init__(self, home_template: Path, mode: str = "acp", hermes_bin: str = "hermes",
                 timeout: float = 900, review_grace: float = 45, max_turns: int = 40,
                 toolsets: str | None = None, label: str | None = None):
        if mode not in ("acp", "oneshot"):
            raise ValueError("mode must be acp or oneshot")
        self.template, self.mode, self.bin = Path(home_template), mode, hermes_bin
        self.timeout, self.review_grace, self.max_turns, self.toolsets = timeout, review_grace, max_turns, toolsets
        self.name = label or f"hermes-{mode}"
        self.client: AcpClient | None = None
        self.session: str | None = None
        self.home: Path | None = None
        self.run_dir: Path | None = None
        self.homes = 0

    def _fresh_home(self) -> None:
        self._stop()
        self.homes += 1
        self.home = self.run_dir / "homes" / f"home-{self.homes:03d}"
        if self.home.exists():
            shutil.rmtree(self.home)
        shutil.copytree(self.template, self.home, symlinks=True)

    def _env(self) -> dict:
        return {**os.environ, "HERMES_HOME": str(self.home)}

    def _stop(self) -> None:
        if self.client:
            self.client.close()
            self.client = None

    def begin_run(self, run_dir):
        self.run_dir = run_dir
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._fresh_home()

    def reset_state(self):
        self._fresh_home()

    def end_run(self):
        self._stop()

    def begin_episode(self, ep: Episode, ws: Path):
        self.session = None
        self.log = self.run_dir / "logs" / f"ep{ep.index:03d}"
        if self.mode == "acp":
            if self.client is None:
                self.client = AcpClient([self.bin, "acp"], self._env(), self.run_dir, self.log.with_suffix(".acp.jsonl"))
                self.client.request("initialize", {"protocolVersion": 1, "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False}}, 120)
            res = self.client.request("session/new", {"cwd": str(ws), "mcpServers": []}, 120)
            self.session = res["sessionId"]

    def attempt(self, ep, ws, message, *, followup, violated=None):
        return self._acp_turn(message) if self.mode == "acp" else self._oneshot_turn(ws, message, followup)

    def _acp_turn(self, message: str) -> dict:
        res = self.client.request("session/prompt", {"sessionId": self.session,
                                                     "prompt": [{"type": "text", "text": message}]}, self.timeout)
        usage: dict = {"stop_reason": res.get("stopReason")}
        _harvest_usage(res, usage)
        _harvest_usage(self.client.take_updates(), usage)
        return usage

    def _oneshot_turn(self, ws: Path, message: str, followup: bool) -> dict:
        argv = [self.bin, "chat", "-q", message, "--format", "stream-json", "--yolo", "--source", "tool",
                "--max-turns", str(self.max_turns)]
        if self.toolsets:
            argv += ["-t", self.toolsets]
        if followup and self.session:
            argv += ["-r", self.session]
        p = subprocess.run(argv, cwd=ws, env=self._env(), capture_output=True, text=True, timeout=self.timeout)
        tag = "retry" if followup else "first"
        self.log.with_suffix(f".{tag}.jsonl").write_text(p.stdout)
        self.log.with_suffix(f".{tag}.stderr").write_text(p.stderr)
        usage: dict = {"exit_code": p.returncode}
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            _harvest_usage(ev, usage)
            sid = ev.get("session_id") or ev.get("sessionId") if isinstance(ev, dict) else None
            if sid:
                self.session = sid
        return usage

    def end_episode(self, ep, ws):
        if self.mode == "acp" and self.review_grace:
            time.sleep(self.review_grace)  # let the background review thread finish writing
        return self.library_stats()

    def library_stats(self) -> dict:
        skills = self.home / "skills"
        files = [p for p in skills.rglob("SKILL.md") if ".archive" not in p.parts] if skills.exists() else []
        archived = list((skills / ".archive").rglob("SKILL.md")) if (skills / ".archive").exists() else []
        mem = [p for p in (self.home / "memories").rglob("*") if p.is_file()] if (self.home / "memories").exists() else []
        # With memory/skills.write_approval on, writes are staged here instead of applied; nothing is learned.
        pending = list((self.home / "pending").rglob("*.json")) if (self.home / "pending").exists() else []
        return {
            "pending_writes": len(pending),
            "skills": len(files),
            "archived_skills": len(archived),
            "skill_bytes": sum(p.stat().st_size for p in files),
            "memory_bytes": sum(p.stat().st_size for p in mem),
            "skill_names": sorted(str(p.parent.relative_to(skills)) for p in files)[:200],
        }

    def run_curator(self, consolidate: bool = False) -> int:
        """`hermes curator run`; the curator judges staleness by wall-clock time, so a short run sees no stale skills."""
        self._stop()
        argv = [self.bin, "curator", "run"] + (["--consolidate"] if consolidate else [])
        p = subprocess.run(argv, env=self._env(), capture_output=True, text=True, timeout=self.timeout)
        (self.run_dir / "logs" / f"curator-{int(time.time())}.log").write_text(p.stdout + p.stderr)
        return p.returncode
