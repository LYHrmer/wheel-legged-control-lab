# Isolated EPA local-gap DSO: exact build provenance

This is the final build explanation for the published source layout `engine_patch/`. It describes the **already built and executed** DSO; no build, model or engine call is performed by this document or by the compact packager. The DSO only runs in a separately guarded and budgeted process. It does not replace an installed MuJoCo library.

The official MuJoCo 3.12.0 `engine_collision_gjk.c` source SHA256 is `0f3ff5b77893132dcc8412d4e96d80f9d625e99cb3009bb7a4505eba16320855`. `src/engine_local_gjk.c` keeps the original copyright/Apache header and, apart from four local symbol aliases, changes exactly one numerical predicate from `upper - lower < tolerance` to `upper_k - lower < tolerance`; compare `provenance/local_kernel_vs_original.diff`. The local point/line helper in `src/engine_local_support.c` comes from fixed upstream `engine_collision_convex.c`. No unrelated model, solver or control gain is changed.

## Actual frozen compiler invocation

The original build used GCC 11.4.0 from cwd `/home/lyh`; the exact argv was:

```sh
rtk proxy cc -std=c11 -O0 -fPIC -shared -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror -I/home/lyh/.local/lib/python3.10/site-packages/mujoco/include -I/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_epa01/sources -I/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01 /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/engine_local_gjk.c /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/engine_local_support.c /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/engine_interpose.c -o /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/build/libepa01_engine.so -L/home/lyh/.local/lib/python3.10/site-packages/mujoco -Wl,-rpath,/home/lyh/.local/lib/python3.10/site-packages/mujoco -l:libmujoco.so.3.12.0 -ldl -lm
```

The resulting file SHA256 is `3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c`. The linked original `libmujoco.so.3.12.0` SHA256 is `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`. The three `provenance/deps_engine_*.d` files were generated independently for the three C translation units; `provenance/build_dependencies.json` records 125 exact build dependency hashes. `nm_defined.txt`, `nm_undefined.txt` and `readelf_dynamic.txt` show wrapper exports, no unresolved old `mjc_ccd`/`mj_step`, the fixed `NEEDED` library and the absolute RUNPATH.

## Relative source mapping for inspection or an isolated rebuild

The public `engine_patch/` tree maps the original `-I` directories and source operands as follows. All numerical, optimization, warning, linker and math flags stay identical; the output path is new.

| Original operand | Public source mapping |
| --- | --- |
| installed `mujoco/include` | explicit `MUJOCO_DIR/include`, with the exact original header/library fingerprints from `build_dependencies.json` |
| `stability_20260923_epa01/sources` | `engine_patch/upstream` |
| `stability_20260923_engine01` | `engine_patch/src` |
| three `engine_local_*.c` / `engine_interpose.c` operands | corresponding `engine_patch/src/*.c` |
| original `build/libepa01_engine.so` | a **new** isolated output directory |
| original `-L` and RUNPATH | the selected byte-identical `MUJOCO_DIR` |

For a fresh output path after verifying the installed library SHA and upstream headers, the path-mapped command is:

```sh
cc -std=c11 -O0 -fPIC -shared -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror \
  -I"$MUJOCO_DIR/include" -I"$PACKAGE_DIR/engine_patch/upstream" -I"$PACKAGE_DIR/engine_patch/src" \
  "$PACKAGE_DIR/engine_patch/src/engine_local_gjk.c" \
  "$PACKAGE_DIR/engine_patch/src/engine_local_support.c" \
  "$PACKAGE_DIR/engine_patch/src/engine_interpose.c" \
  -o "$OUTPUT_DIR/libepa01_engine.so" \
  -L"$MUJOCO_DIR" -Wl,-rpath,"$MUJOCO_DIR" \
  -l:libmujoco.so.3.12.0 -ldl -lm
```

This maps source operands and preserves **every compiler/linker flag** from the executed command; it is not a promise of byte-identical output outside the original absolute RUNPATH/compiler environment. A different installation or rebuilt DSO requires a new hash, ELF/GOT proof and physical readiness qualification. `provenance/engine_binding.py` pins the original ELF/GOT addresses and is included as inspection evidence; it is not a cross-platform or cross-version loader. An optional copied original DSO is an audit artifact, never an automatic deployment. The complete MuJoCo copyright and third-party licenses are included beside this file.
