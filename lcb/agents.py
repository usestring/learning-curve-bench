"""Agent adapters. Reference agents are deterministic and free; they validate the metrics.

An adapter receives the task text (and, on a retry, the feedback text) for a workspace and
must leave its output in that workspace. Real agents see only text. Reference agents also
get the violated rule ids, because their job is to model a kind of memory exactly, not to
read prose.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .families import FAMILIES, apply_solution
from .schedule import Episode


class Agent:
    name = "agent"

    def begin_run(self, run_dir: Path) -> None: ...

    def reset_state(self) -> None:
        """Forget everything learned. The runner calls this per episode in the frozen condition."""

    def begin_episode(self, ep: Episode, ws: Path) -> None: ...

    def attempt(self, ep: Episode, ws: Path, message: str, *, followup: bool,
                violated: list[str] | None = None) -> dict:
        """Do one turn. Returns usage, e.g. {"input_tokens": .., "output_tokens": .., "cost_usd": ..}."""
        raise NotImplementedError

    def end_episode(self, ep: Episode, ws: Path) -> dict:
        """Returns library stats to record (skill counts and the like)."""
        return {}

    def end_run(self) -> None: ...


def _rule_slots(family: str) -> dict[str, str]:
    fam = FAMILIES[family]
    return {r.id: r.slot for e in (0, 1) for r in fam.rules(e)}


class NullAgent(Agent):
    """Never learns; follows generic defaults every time. The floor."""

    name = "null"

    def attempt(self, ep, ws, message, *, followup, violated=None):
        known = set(violated or []) if followup else set()
        apply_solution(FAMILIES[ep.family].solve(ep.variant, known), ws)
        return {}


class OracleAgent(Agent):
    """Knows every rule that is currently in force. The ceiling."""

    name = "oracle"

    def attempt(self, ep, ws, message, *, followup, violated=None):
        apply_solution(FAMILIES[ep.family].solve(ep.variant, set(FAMILIES[ep.family].rule_ids(ep.epoch))), ws)
        return {}


class NotebookAgent(Agent):
    """Writes each corrected rule into a per-slot notebook and applies the notebook next time.

    stale=True never overwrites a slot it already holds: it fixes the retry it is in but keeps
    the old rule, like a skill that never gets patched after the world changes.
    capacity=N keeps only the N most recently used rules, which produces regressions.
    """

    def __init__(self, stale: bool = False, capacity: int | None = None):
        self.stale, self.capacity = stale, capacity
        self.name = "stale-notebook" if stale else f"forgetful-notebook-{capacity}" if capacity else "notebook"
        self.notes: OrderedDict[tuple[str, str], str] = OrderedDict()

    def reset_state(self):
        self.notes.clear()

    def _known(self, family):
        known = set()
        for (fam, slot), rid in list(self.notes.items()):
            if fam == family:
                known.add(rid)
                self.notes.move_to_end((fam, slot))
        return known

    def attempt(self, ep, ws, message, *, followup, violated=None):
        slots = _rule_slots(ep.family)
        known = self._known(ep.family)
        if followup and violated:
            for rid in violated:
                if rid not in slots:
                    continue
                key = (ep.family, slots[rid])
                known = {k for k in known if slots.get(k) != slots[rid]} | {rid}
                if not (self.stale and key in self.notes):
                    self.notes[key] = rid
                    self.notes.move_to_end(key)
            while self.capacity is not None and len(self.notes) > self.capacity:
                self.notes.popitem(last=False)
        apply_solution(FAMILIES[ep.family].solve(ep.variant, known), ws)
        return {}


REFERENCE_AGENTS = {
    "null": NullAgent,
    "oracle": OracleAgent,
    "notebook": NotebookAgent,
    "stale-notebook": lambda: NotebookAgent(stale=True),
    "forgetful-notebook": lambda: NotebookAgent(capacity=4),
}
