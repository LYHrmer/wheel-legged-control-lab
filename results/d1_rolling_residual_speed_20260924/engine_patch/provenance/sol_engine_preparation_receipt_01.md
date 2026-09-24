# Isolated engine C/D preparation receipt

2026-09-23. Implemented by GPT-6-sol. This receipt documents static preparation only; root is the sole owner of C/D engine execution.

The clean local kernel was generated from fixed official MuJoCo 3.12.0 `engine_collision_gjk.c` SHA `0f3ff5b77893132dcc8412d4e96d80f9d625e99cb3009bb7a4505eba16320855`. `build/local_kernel_vs_original.diff` contains four symbol-renaming macros and the sole numerical edit `upper - lower` to `upper_k - lower`. The point/line helper arithmetic is copied from fixed original `engine_collision_convex.c`, with no trace globals. The DSO exports original-ABI `mjc_ccd` and `mj_step`, delegates CCD to the isolated kernel, and atomically guards construction/control steps. Python `EngineGuard` proves the four fixed actual JUMP_SLOT targets, ELF hashes, `RTLD_NEXT` original step identity, and first CCD/TryCompile caller offsets.

Final files (SHA-256):

| File | SHA-256 |
|---|---|
| `prepare_engine_kernel_01.py` | `54bd451527bc1eedcfa9de9c99d0fe4e7d09b31525f44034323740da81659af2` |
| `freeze_engine_build_inputs_01.py` | `4fae5d37e56cc0b2cdb8d29ca09c2649523e4ebfc9b191756677a7d6e5f0d7ef` |
| `engine_local_gjk.c` | `26d2b35c42390915061dd99a788652357d4e43fa40a15307c725b86aa8ee51ef` |
| `engine_local_support.c` | `3526fc64b73e6b231a761d59188049f2cef69a0b6b46d50cdeffc2704c15c172` |
| `engine_interpose.c` | `bc09b8344457df62778c98f36807740da4ef66d8b9f5c1a60af8a7a133f541d5` |
| `engine_interpose.h` | `99ac0297569f2e2eaf0e2878d531a3f1b107825d09a75a4a94ef8f3783ec5bc2` |
| `engine_binding.py` | `e0ab759d170a330972fb9d195c89711991484198717874d661d1431b91de6ce1` |
| `build/libepa01_engine.so` | `3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c` |
| `build/build_dependencies.json` | `82468459a24c9c6f2c2ee93d6eb6c01f2f78459d63f99da63802ddefbaf9e881` |

Exact static build argv (cwd `/home/lyh`):

```sh
rtk proxy cc -std=c11 -O0 -fPIC -shared -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror -I/home/lyh/.local/lib/python3.10/site-packages/mujoco/include -I/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_epa01/sources -I/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01 /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/engine_local_gjk.c /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/engine_local_support.c /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/engine_interpose.c -o /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923_engine01/build/libepa01_engine.so -L/home/lyh/.local/lib/python3.10/site-packages/mujoco -Wl,-rpath,/home/lyh/.local/lib/python3.10/site-packages/mujoco -l:libmujoco.so.3.12.0 -ldl -lm
```

GCC is Ubuntu 11.4.0. Three independent `-M` dependency passes, saved as `build/deps_engine_*.d`, contributed to `build/build_dependencies.json` (125 files). `build/nm_defined.txt` confirms exported wrappers and local kernel; `build/nm_undefined.txt` has no unresolved `mjc_ccd` or `mj_step`; `build/readelf_dynamic.txt` records fixed `NEEDED libmujoco.so.3.12.0` and installed-library RUNPATH. Fixed original relocation evidence is in `build/original_relocations.txt`; extension JUMP_SLOT offsets are bound to known SHA in `engine_binding.py`.

All three C files passed `-fsyntax-only -Wall -Wextra -Werror` and linked. The three pure Python files passed repository-config Ruff. No MuJoCo Python import, DSO load, CCD call, model construction, step, dynamic test, or C/D runner execution was performed by Sol. C/D remain subject to Astra review and root's exclusive execution budgets.
