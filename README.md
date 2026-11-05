# Learning Curve Bench

Does an agent get better at recurring work from feedback, keep what it learned, and notice when a
convention changes? Learning Curve Bench measures first attempts across repeated tasks, with
a memory-wiped control, feedback-free probes, and conventions that change during the run.

Python 3.10+, standard library only. No API key is needed for offline validation.

```bash
git clone https://github.com/usestring/learning-curve-bench.git
cd learning-curve-bench
```

Run the commands below from the repository root.

```bash
python3 -m lcb validate --out runs/validation  # reference agents, offline and free
python3 -m unittest discover -s tests   # verifier, schedule, metric and adapter tests
python3 -m lcb show --seed 0            # the episode stream
python3 -m lcb show --seed 0 --index 7  # one episode's prompt, files and full feedback text
python3 -m lcb report runs/             # summarize any set of runs
```

## Design

A run is a seeded stream of 40 episodes drawn from five task families. Each learnable family has
house rules the agent can only learn from feedback. The prompt never states them.

| Family | Task | House rules (one independent check each) |
|---|---|---|
| `release_notes` | changelog → `out/RELEASE_NOTES.md` | header line; Fixes before Features; drop chore/docs; ticket format (**drifts**) |
| `acmectl` | deploy with a fake CLI that logs argv | take the lock first; `v` version prefix; env flag `--env` → `--target` (**drifts**; v2 CLI warns on `--env`) |
| `csv_cleanup` | `raw.csv` → `out/clean.csv` | amounts in integer cents; dedupe by email; DD/MM dates → ISO; fixed column set and order |
| `triage` | `incidents.json` → `out/triage.json` | three binding priority rules, at least one instance of each per variant |
| `log_count` | count lines (control, no rules) | none: shows that capability and harness are not the bottleneck |

Episode loop:

1. The agent gets a fresh workspace and the task. **Its first attempt is scored before any
   feedback.** Every headline metric uses only first attempts, so it measures what was carried
   over from earlier episodes.
2. If any check fails and the episode is not a probe, the agent gets feedback naming the violated
   rules and one retry. The retry is reported as `feedback_uptake` and is never mixed into the
   learning numbers.
3. Every third occurrence of a family is a **probe** with no feedback at all.
4. At 60% of the stream, two rules change silently (**drift**). Feedback after drift states the new
   rule only.

Conditions, compared paired by seed:

- `learning`: one persistent agent state for the whole stream.
- `frozen`: state reset before every episode. Same model and harness, no carry-over.

The headline is the **learning gain**: post-first-occurrence first-try success, learning minus
frozen.

## Metrics

| Metric | Meaning |
|---|---|
| `first_try_success`, `aulc` | solved / mean check score on first attempts (learnable families) |
| `post_first_success` | same, excluding each family's first occurrence (where nothing can be known yet) |
| `probe_success` | first-try success on no-feedback probes |
| `feedback_uptake` | retry success: can the agent follow feedback at all (in-context, not learning) |
| `regression_rate` | a rule satisfied earlier, violated later, with no rule change in between |
| `drift_unrepaired`, `drift_repair_latency` | changed rules never fixed / post-drift occurrences until fixed |
| `control_success` | must stay near 1; if not, results reflect the harness, not learning |
| `cost_per_first_try_solve` | total spend / first-try solves (when the agent reports usage) |

Reference agents establish the scale: `null` (never learns), `oracle` (always knows the current
rules), `notebook` (writes down every feedback item), `stale-notebook` (never overwrites an old rule)
and `forgetful-notebook-4` (LRU memory of four rules). A metric that cannot order these is broken;
the tests require that it can.

## Running Hermes Agent

`lcb/hermes.py` drives [Hermes Agent](https://github.com/NousResearch/hermes-agent). **It has only
been tested against fake Hermes binaries.** There are no live Hermes results yet. The reference
agents validate the harness; they are not evidence that a language model learns.
See [VALIDATION.md](VALIDATION.md) for reproduced results and adapter limitations.

Running a real model incurs API charges. Use a dedicated account or spending limit and start
with a short run in an isolated environment: the adapter automatically approves tool permissions.

```bash
# A template HERMES_HOME holding only config.yaml and .env (model, provider, key). Kept outside git.
python3 -m lcb run --agent hermes --hermes-home-template ~/lcb-hermes-home \
  --condition learning --episodes 6 --seeds 0 --out runs/
python3 -m lcb run --agent hermes --hermes-home-template ~/lcb-hermes-home \
  --condition frozen --episodes 6 --seeds 0 --out runs/
python3 -m lcb report runs/
```

- `--hermes-mode acp` (default) keeps one `hermes acp` process alive, so the after-turn background
  review can write skills and memory. Each episode is a new ACP session, and permission requests
  are auto-allowed. `--review-grace` waits after each episode for the review to finish.
- `--hermes-mode oneshot` runs `hermes chat -q` per attempt. Hermes disables background review for
  finite one-shot turns, so this measures only in-turn saving. Use `--label` to keep the runs apart.
- `--curator-every N [--curator-consolidate]` runs `hermes curator run` between episodes.
  Staleness is judged by wall-clock time, so a run of a few hours archives nothing. Consolidation
  uses LLM calls.
- Per-episode `library` records SKILL.md count and bytes, archived skills, memory bytes, and
  `pending_writes`. A non-zero `pending_writes` means `write_approval` is on and nothing is being
  learned.

A six-episode run checks connectivity and record creation. Use the default 40 episodes and the
same seeds in both conditions for learning comparisons. Keep the model, configuration, Hermes
version, and harness mode fixed. Record the exact Hermes revision with any published results.

Run directories include copies of the home template and may contain credentials. Keep them
private; see [corpus.md](corpus.md) before sharing any output.

## License

[MIT](LICENSE).
