# Isolated MuJoCo CCD patch: source package draft

Status: **draft while the sole rolling RL process is running**. This file is outside the repository and outside the frozen run inputs. It is a packaging specification, not a claim that the RL run passed or that a patched MuJoCo installation has been released. Do not edit the frozen DSO, repository scripts, run records, or old score archives to publish it.

## Scope and provenance

The candidate is a Linux x86-64 MuJoCo 3.12.0 `LD_PRELOAD` DSO. It replaces only the private CCD entry used by the fixed engine and wraps `mj_step` to count and restrict the experiment's dynamic calls. It does **not** overwrite the installed MuJoCo library. The local GJK source is copied from the pinned official `engine_collision_gjk.c` (SHA256 `0f3ff5b77893132dcc8412d4e96d80f9d625e99cb3009bb7a4505eba16320855`); the only numerical source change is the EPA termination predicate `upper - lower < tolerance` → `upper_k - lower < tolerance`. Four macro aliases rename local entry points so the wrapper exports the original ABI. Preserve the source's copyright and Apache 2.0 header.

The frozen candidate DSO is SHA256 `3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c`. A/B source-local 40-query evidence, C1 binding evidence, and D two-case 0.2 m/s physical readiness evidence remain separate records. The current rolling RL/speed process is an additional, independently budgeted experiment; its outcome must be filled from its final receipts only after root closes it. Do not replace an old invalid score with a new engine's result.

## Proposed public source layout

Place a new directory under the existing contact forensics result package, for example `engine_patch_20260923_01/`. Keep this draft external until root decides the final publication content and checks the one process result.

| Package path | Read-only source | SHA256 |
| --- | --- | --- |
| `src/engine_local_gjk.c` | `stability_20260923_engine01/engine_local_gjk.c` | `26d2b35c42390915061dd99a788652357d4e43fa40a15307c725b86aa8ee51ef` |
| `src/engine_local_support.c` | `stability_20260923_engine01/engine_local_support.c` | `3526fc64b73e6b231a761d59188049f2cef69a0b6b46d50cdeffc2704c15c172` |
| `src/engine_interpose.c` | `stability_20260923_engine01/engine_interpose.c` | `bc09b8344457df62778c98f36807740da4ef66d8b9f5c1a60af8a7a133f541d5` |
| `src/engine_interpose.h` | `stability_20260923_engine01/engine_interpose.h` | `99ac0297569f2e2eaf0e2878d531a3f1b107825d09a75a4a94ef8f3783ec5bc2` |
| `upstream/engine/engine_collision_gjk.c` | `stability_20260923_epa01/sources/engine/engine_collision_gjk.c` | `0f3ff5b77893132dcc8412d4e96d80f9d625e99cb3009bb7a4505eba16320855` |
| `upstream/engine/engine_collision_gjk.h` | same `sources/engine/` | `756622585c9b5d1c5f824eb8e7d69c8c8393addd74733cfddbb01321ff3a219e` |
| `upstream/engine/engine_collision_convex.h` | same `sources/engine/` | `3df5241f900dd9c01565c857191eeb740104b09f75a11af56500b73fbaefddfe` |
| `upstream/engine/engine_inline.h` | same `sources/engine/` | `c1aef20e999a7f5a9e11b4f8aa05ee66179cb3ef77f432b6598980fb70a5c669` |
| `upstream/engine/engine_util_blas.h` | same `sources/engine/` | `8f4cc330efe776e33b1cbc12a09ea5339659b239356606c80a1e9398576cb269` |
| `upstream/engine/engine_util_errmem.h` | same `sources/engine/` | `44a2658081d675f6a918893088fc06478f9083457bd6cf8a93d2c8f03e75bd9b` |
| `provenance/local_kernel_vs_original.diff` | `stability_20260923_engine01/build/local_kernel_vs_original.diff` | hash when copied |
| `provenance/build_dependencies.json` | `stability_20260923_engine01/build/build_dependencies.json` | `82468459a24c9c6f2c2ee93d6eb6c01f2f78459d63f99da63802ddefbaf9e881` |
| `provenance/sol_engine_preparation_receipt_01.md` | `stability_20260923_engine01/sol_engine_preparation_receipt_01.md` | hash when copied |
| `provenance/engine_binding.py` | `stability_20260923_engine01/engine_binding.py` | `e0ab759d170a330972fb9d195c89711991484198717874d661d1431b91de6ce1` |
| `LICENSE` and `LICENSES_THIRD_PARTY.md` | existing result package `sources/MUJOCO_LICENSE` and `MUJOCO_LICENSES_THIRD_PARTY.md` | hash when copied |

Include the exact source URL/tag and download SHA manifest already saved under `stability_20260923_epa01/sources/manifest.json`. Include DSO `build/nm_defined.txt`, `nm_undefined.txt`, `readelf_dynamic.txt`, and three `deps_engine_*.d` files as build provenance; the merged dependency JSON alone does not recreate compiler argv. The source package must retain attribution for copied `engine_local_support.c` point/line helper arithmetic from official `engine_collision_convex.c`; include that upstream `.c` as a comparison reference if space permits.

## Build recipe to document, not run during the active RL process

Inputs: a byte-identical official MuJoCo 3.12.0 installation with headers and `libmujoco.so.3.12.0` SHA256 `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`, GCC 11.4.0, C11, and the package's `src/` and `upstream/` trees. The reviewed local build used `-O0 -fPIC -shared -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror`, linked `-l:libmujoco.so.3.12.0 -ldl -lm`, and set RUNPATH to the fixed installed library directory. A public recipe should accept an explicit `MUJOCO_DIR` and refuse to build if its library SHA differs from this original binary fingerprint. It should emit to a new output path, never into the installed MuJoCo directory or over the frozen candidate.

Illustrative relative invocation after packaging (adjust paths only; do not alter flags or source math):

```sh
cc -std=c11 -O0 -fPIC -shared -fno-fast-math -ffp-contract=off \
  -Wall -Wextra -Werror \
  -I"$MUJOCO_DIR/include" -I"$PACKAGE_DIR/upstream" -I"$PACKAGE_DIR/src" \
  "$PACKAGE_DIR/src/engine_local_gjk.c" \
  "$PACKAGE_DIR/src/engine_local_support.c" \
  "$PACKAGE_DIR/src/engine_interpose.c" \
  -o "$OUTPUT_DIR/libepa01_engine.so" \
  -L"$MUJOCO_DIR" -Wl,-rpath,"$MUJOCO_DIR" \
  -l:libmujoco.so.3.12.0 -ldl -lm
```

The original DSO SHA is an exact artifact identity for the original absolute RUNPATH and compiler environment. A build elsewhere may differ bytewise due to path/toolchain even when source math is equivalent; its new binary must receive a new SHA and a new binding/readiness qualification. The recorded `engine_binding.py` intentionally pins the original ELF hashes and GOT offsets and must not be presented as a generic cross-version loader. The wrapper is experimental and must be loaded only for a specifically budgeted process with the approved phase/target guard; no default installation, global preload, or production/hardware use follows from these simulator trials.

## Publication linkage after the running process closes

Root should add only verified final run outcomes: sole-process activation and closure, frozen input manifest, actual compiler/control/native/CCD counters, source hash comparison, PPO update/checkpoint/reload proof, full eight-case score table and speed deltas, and exact nonqualification reasons if any. Mark the final status from those receipts. The present source package can support reproduction and inspection of the patch regardless of whether the bounded RL claim passes; do not describe an attempted or incomplete run as a successful RL/speed result.
