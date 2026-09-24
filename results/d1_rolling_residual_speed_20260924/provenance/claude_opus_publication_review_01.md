# Offline publication-packaging review 01 (Claude Opus, static source only)

Scope: static read of the packager, the figure script, the offline readback, and the
closed run_01 records. No shell, engine, model, policy, test or network call was made;
no existing file was edited. The root-owned physics process is still executing held-out
evaluations, so **no physics or qualification statement is made here**.

Sources actually read (verbatim paths):

- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/publication_staging/build_compact_publication_01.py`
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/publication_staging/plot_saved_rolling_trajectories_01.py`
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/publication_staging/engine_patch_build_01.md`
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/root_readback_02.py`
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/root_execution_freeze_02.json` (first 857 of 1884 lines)
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/run_01/root_launcher_receipt.json`
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/run_01/study_receipt.json`
- `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/run_01_boundary_closure.json`
- `/home/lyh/wheel-legged-control-lab/scripts/d1_rolling_residual_task.py`
- `/home/lyh/wheel-legged-control-lab/scripts/d1_rolling_residual_checkpoint.py`
- `/home/lyh/wheel-legged-control-lab/scripts/evaluate_d1_rolling_residual.py`

Note: the packager is being edited by Sol in parallel; every quote below is the text as
read in this session. Line numbers refer to that state.

---

## Blockers

### B1 — The mandatory independent readback artifacts are not in the package at all

`run_01/root_launcher_receipt.json` states the claim policy itself:

```json
"all_claims_require_independent_training_and_evaluation_readback": true,
```

`root_readback_02.py:143-149` produces exactly those artifacts:

```python
write(W/'run_02_archive_manifest_01.json',{'scope':'all closed run_02 files; no files omitted','files':manifest})
write(W/'root_run_02_readback_01.json',{'passed':True,'scope':'offline raw training/action/update/hash/paired-state/native-chain/control-torque reconciliation; no engine or policy import',
```

`build_compact_publication_01.py:232-279` has **no** entry for `root_run_02_readback_01.json`,
for `run_02_archive_manifest_01.json`, or for `root_readback_02.py` itself (only
`build_compact_publication_01.py` and `plot_saved_rolling_trajectories_01.py` are placed
under `provenance/`, lines 275-278). The README written at line 106-108 then reports
outcomes read straight from the study receipt, i.e. the package asserts results whose
only independent reconciliation is left outside the package.

Severity: blocker — a mandatory artifact class is silently omitted.
Minimal correction: add to `package_map`

```python
"evidence/rl/root_run_02_readback_01.json": rl / "root_run_02_readback_01.json",
"evidence/rl/run_02_archive_manifest_01.json": rl / "run_02_archive_manifest_01.json",
"provenance/root_readback_02.py": rl / "root_readback_02.py",
```

and hard-fail unless the readback JSON has `passed is True`, its `study_qualified` equals
`study["passed"]`, and its `actual_controls` equals `study["recorded_control_intervals"]`.

### B2 — The crashed run_01 attempt is packaged as unlabelled evidence while the README describes a single run

`build_compact_publication_01.py:252-255` hardcodes:

```python
"evidence/rl/run_01_boundary_closure.json": rl / "run_01_boundary_closure.json",
"evidence/rl/run_01/study_receipt.json": rl / "run_01/study_receipt.json",
"evidence/rl/run_01/root_launcher_receipt.json":
    rl / "run_01/root_launcher_receipt.json",
```

run_01 is not the packaged run. `root_execution_freeze_02.json:10` pins

```json
"output_directory": "/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_rl01/run_02",
```

and the packager enforces that at line 174 (`freeze["output_directory"] != str(run)`), so
`--run` must be run_02. run_01 is a closed non-physical failure:
`run_01/root_launcher_receipt.json` has `"exit_code": 1`, `run_01/study_receipt.json` has
`"failure": {"message": "LD_LIBRARY_PATH must be empty", ...}`, `"passed": false`, and
`run_01_boundary_closure.json` reports `"completed_training_controls": 0`.

The package therefore ships `evidence/rl/root_launcher_receipt.json` (run_02) next to
`evidence/rl/run_01/root_launcher_receipt.json` (exit 1) with identical file names,
while `result_readme` (lines 98-102) says only:

```
One 65,536-control PPO run, one final checkpoint if completed, and up to eight
```

Nothing in the README explains the second, contradictory receipt pair.

Severity: blocker — as packaged this misstates the evidence (reads as either a hidden
retry or an unexplained failed receipt), and the freeze/launcher records show the prior
attempt consumed the full reservation.
Minimal correction: re-key the three entries to a self-describing prefix, e.g.
`evidence/rl/prior_closed_attempt_run_01/...`, and add one README line stating that
attempt run_01 closed at environment-bootstrap with zero completed controls
(`LD_LIBRARY_PATH must be empty`), was not retried, and contributed no physics.

### B3 — README qualification and `publication_manifest.json` qualification can contradict each other; nonzero `exit_code` is not gated

Guard, lines 168-171:

```python
if launcher.get("process_attempts") != 1 or launcher.get("error") is not None:
    raise RuntimeError("root process did not close as one completed attempt")
if launcher.get("post_execution_hash_mismatches"):
    raise RuntimeError("root reported changed frozen sources")
```

`exit_code` is never checked here — and `run_01/root_launcher_receipt.json` proves a
receipt can carry `"exit_code": 1` with `"error": null` and `"process_attempts": 1`, i.e.
this guard passes for a failed launch. Downstream the two qualification statements use
different predicates:

```python
qualified = bool(study.get("passed") is True and launcher.get("exit_code") == 0
                 and not launcher.get("post_execution_hash_mismatches"))          # line 84-85
...
"qualification_reported_from_closed_receipts": bool(study.get("passed") is True),  # line 337
```

If `study["passed"]` is True while `exit_code != 0`, README prints **not qualified** and
`publication_manifest.json` records qualification **true**, inside one package.

Severity: blocker for evidence consistency.
Minimal correction: compute `qualified` once before writing anything, pass it to both
writers, and raise if `study.get("passed") is True and launcher.get("exit_code") != 0`.

---

## Medium findings

### M4 — The non-qualification reason is never published

`study_receipt.json` carries a `failure` object (`run_01/study_receipt.json` shows the
shape). `result_readme` reports `passed`, `recorded_control_intervals`,
`actual_counts_consistent` and receipt presence (lines 106-108) but never `failure`, so a
non-qualified package states the verdict without the cause. `root_readback_02.py:39`
asserts on exactly that field (`assert receipt['failure'] is None`).
Minimal correction: append `f"- Study failure: \`{study.get('failure')}\`."` whenever it is
not `None`.

### M5 — Checkpoint verification ignores two hashes the run already recorded

Lines 296-300 check only the model hash:

```python
sidecar = load_json(checkpoint / "model.metadata.json")
if sidecar.get("model_sha256") != digest(checkpoint / "model.zip")["sha256"]:
```

`d1_rolling_residual_checkpoint.py:107` also writes `reload_probe_sha256`, and
`root_readback_02.py:52` reconciles `study['training']['checkpoint_metadata_sha256']`. The
packager copies `model.metadata.json` and `reload_probe.npz` (line 301) without binding
either to those recorded digests, so a post-hoc edit of the sidecar body (e.g.
`num_timesteps`, `recorded_episode`) or a swapped probe still packages cleanly.
Minimal correction: also require
`digest(checkpoint / "model.metadata.json")["sha256"] == study["training"]["checkpoint_metadata_sha256"]`
and `sidecar["reload_probe_sha256"] == digest(checkpoint / "reload_probe.npz")["sha256"]`.

### M6 — `--figures` is accepted with no binding to `--run`, and empty figures are publishable

Packager lines 319-323 copy `speed_tracking.png`, `box_pitch.png`,
`shared_leg_residual.png`, `figure_receipt.json` from any directory that contains those
four names. `figure_receipt.json` (plot script lines 178-189) records
`study_passed_from_receipt`, per-case `completed_control` and PNG byte counts — but no run
path and no endpoint/trace hash, so a stale figure set cannot be distinguished. Separately,
`plot()` does not require any case to exist: `_read_case` returns `None` for a missing
directory (lines 48-50) and every plotting loop does `if case is None: continue`, so three
axis-only PNGs plus a `figure_receipt.json` with an empty `"sources"` map are a valid
output.
Minimal correction: record `"run": str(run)` plus the per-case `endpoints.jsonl.gz` /
`trace.jsonl.gz` sha256 in the figure receipt; raise in `plot()` if no case was read; and
in the packager verify `figure_receipt["run"] == str(run)` before copying.

### M7 — The compact package cannot support the paired zero/final claim it makes

README line 100-101 advertises "fixed paired zero/final-policy held-out cases". The only
machine-checkable proof of the paired initial state is `states.npz`, used by
`root_readback_02.py:81-82`:

```python
if case['actor']=='zero': paired[pair]=initial
else:
    for k in initial: assert same(initial[k],paired[pair][k]),(name,k)
```

The per-case copy loop (lines 307-311) takes only `score.json`, `receipt.json`,
`episode_metadata.json`, `geometry_manifest.json`; `states.npz` is excluded, and
`evaluate_d1_rolling_residual.py:216` puts only the self-asserted boolean
`"paired_initial_exact"` into the summary. So verification case 4 below is not executable
against the compact archive unless `--full-archive-manifest` is used.
Minimal correction: either make `--full-archive-manifest` mandatory when the run is
qualified, or add a small per-case `paired_initial.json` holding only row 0 of the five
`PAIR_FIELDS`, and say plainly in the README that the paired check is delegated to the
retained workspace records plus the readback receipt.

### M8 — A single missing source leaves a partial destination that blocks a clean retry

`destination.mkdir(parents=True, exist_ok=False)` runs at line 180, `copy_map` at line 318,
and `args.figures.resolve(strict=True)` only at line 320. `copy_one` raises
`FileNotFoundError(source)` on the first absent input (line 42), so a typo or an
unfinished upstream artifact leaves a half-populated directory that the next invocation
refuses to reuse.
Minimal correction: before `mkdir`, resolve `--figures` and raise once with the full list
`[p for p in package_map.values() if not p.is_file()]`.

---

## Improvements (non-blocking)

- Line 349 prints `"published_files": len(package_map)`, which excludes the compact
  trajectory JSONs (lines 314-317), `README.md`, `full_run_manifest.json` and
  `publication_manifest.json`; the number understates the package. Use the manifest length.
- `plot_saved_rolling_trajectories_01.py:170` uses a literal `40.0` for the mm conversion.
  This is numerically correct: `d1_rolling_residual_task.py:27-29` gives
  `LEG_NORMALIZED_SCALE = 0.25`, `ORIGINAL_LEG_EXTENSION_SCALE_M = 0.04`, so applied
  `action[0] ∈ [-0.25, 0.25]` maps to ±10 mm, consistent with `ax.set_ylim(-10.8, 10.8)`.
  Import the constant instead of duplicating the factor.
- `evaluate_d1_rolling_residual.py:217` also writes `{name}.case.json` per case (small,
  contains `paired_initial_exact`); the packager skips it because the loop at line 307
  only iterates directories. Worth adding.

## Verified correct (no change needed)

- `verify_freeze` (lines 50-56) compares `digest(path) != expected` as whole dicts; the
  actual freeze entries carry exactly `{"sha256", "bytes"}` (including the key-order-swapped
  `d1_rolling_engine_runtime.py` entry), so dict equality holds and this is not a false gate.
- Figure case naming `f"speed{round(1000 * speed):03d}_{terrain}_{actor}"` matches
  `evaluate_d1_rolling_residual.py:205` exactly, and `receipt["completed_control_intervals"]`
  exists (`evaluate_d1_rolling_residual.py:146`).
- `len(endpoints) == count + 1`, `len(trace) == count` matches the evaluation writer
  (endpoint at tick 0 plus one per completed control).
- The decimation set in `compact_endpoint_rows` retains ticks 275 and 875, the edges of the
  scoring window reconciled at `root_readback_02.py:127-129`.
- `--run` is effectively pinned to run_02 by the line 174 freeze check.

## What cannot be verified from static source

- Anything about run_02: the study/launcher receipts, the eight case directories,
  `evaluation_summary.json`, and the final checkpoint are still being produced by the
  root-owned process. The packager correctly refuses to run before closure
  (`load_json(run / "study_receipt.json")` at line 166 raises while absent). No pass/fail
  physics statement is made here.
- Existence and content of package_map inputs I did not open: the `stability_20260923_engine01`
  and `stability_20260923_epa01` build/provenance files, the `astra_*` review documents,
  `next_contract.md` files, and `tests/` sources and fixtures. Verified present in this
  session: `engine_patch_build_01.md`, `pure_test_round_1_receipt.json`,
  `results/.../sources/MUJOCO_LICENSE`, and the three run_01/closure records quoted above.
  Lines 858-1884 of `root_execution_freeze_02.json` were not read.
- Whether copies, digests, gzip decode, figure rendering or manifest writing actually
  succeed — no execution was performed.
- Sol's in-flight edits to `build_compact_publication_01.py`; re-check the quoted lines
  against the current file before applying any correction.
