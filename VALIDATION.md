# Validation

The deterministic reference agents test whether the metrics recover known learning, forgetting,
and stale-rule behavior. They have direct access to rule identifiers; they are harness controls,
not language-model results.

Reproduce from the repository root:

```bash
python3 -W error::ResourceWarning -m unittest discover -s tests
python3 -m lcb validate --seeds 0 1 2
```

The suite checks independent rule verifiers, both convention epochs, schedule reproducibility,
metric ordering, and the Hermes adapter against fake ACP and one-shot executables.

## Reference results

Three seeds, 40 episodes per condition; intervals are 95% bootstrap intervals over paired seeds.

| Agent | Post-first success | Regression rate | Unrepaired drifts | Gain over frozen |
| --- | ---: | ---: | ---: | ---: |
| oracle | 1.00 | 0.00 | 0 | 0.00 |
| notebook | 0.88 | 0.00 | 0 | 0.88 [0.86, 0.89] |
| stale-notebook | 0.76 | 0.00 | 2 | 0.76 [0.75, 0.79] |
| forgetful-notebook-4 | 0.12 | 0.79 | 1.33 | 0.12 [0.11, 0.14] |
| null | 0.00 | undefined | 2 | 0.00 |

Control-family success is 1.00 for every reference agent. The oracle has zero learning gain
because it already knows every rule in both conditions. The null agent has no regression rate
because it never satisfies a hidden rule on a first attempt.

## Live adapter limits

No live Hermes run has been completed. The adapter is tested only with fake binaries. A live run
must check whether the installed version honors ACP workspace directories, where it reports usage
and session identifiers, and whether the review grace period allows background writes to finish.
Report missing usage as unknown rather than treating it as zero cost.

ACP keeps one process alive across episodes so background review can run. One-shot mode measures
in-turn saving; it does not exercise background review. Memory-wiped runs reset the copied home
before each episode. `pending_writes` indicates staged writes that have not entered active memory.
Context-only and preinstalled-skill comparisons have not been validated.

A preliminary estimate is $0.30–0.60 per episode at model prices of $3 per million input tokens
and $15 per million output tokens. This estimate assumes one or two turns per episode and a
background review, and has not been measured. Six episodes cost roughly $1.80–3.60 per condition;
paired runs can exceed $5. Set a spending limit and use measured usage for larger experiments.
