"""The episode stream: which family recurs when, where conventions drift, which episodes are probes."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .families import FAMILIES, Variant


@dataclass
class Episode:
    index: int
    family: str
    occurrence: int  # 0 = first time this family appears in the stream
    epoch: int  # 0 before the drift point, 1 after
    probe: bool  # probes get no feedback, so they measure retained knowledge only
    variant: Variant


def build(seed: int, episodes: int = 40, drift_at: float = 0.6, probe_every: int = 3,
          families: list[str] | None = None) -> list[Episode]:
    """Families recur in shuffled blocks, so each appears about episodes/len(families) times.

    Every `probe_every`-th occurrence of a family (never the first) is a probe. From index
    `drift_at * episodes` on, drifting families switch to their epoch-1 rules without notice.
    """
    names = families or list(FAMILIES)
    order_rng = random.Random(f"order:{seed}")
    order: list[str] = []
    while len(order) < episodes:
        block = list(names)
        order_rng.shuffle(block)
        order += block
    drift_index = int(drift_at * episodes)
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(order[:episodes]):
        occ = seen.get(name, 0)
        seen[name] = occ + 1
        epoch = 1 if i >= drift_index else 0
        v = FAMILIES[name].generate(random.Random(f"variant:{seed}:{i}"), epoch)
        out.append(Episode(i, name, occ, epoch, occ > 0 and occ % probe_every == probe_every - 1, v))
    return out
