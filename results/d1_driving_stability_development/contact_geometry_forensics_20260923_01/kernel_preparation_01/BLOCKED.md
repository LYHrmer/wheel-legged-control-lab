# Kernel preparation status

2026-09-23 · Implemented by actual GPT-6-sol under GPT-6-astra's source review.

The generated header, XML, and input JSON are pure copies/derivations of the frozen real fixture. The C probe is a static-review draft; its `main` exits with status 3 before any MuJoCo call. It has no `mj_loadXML` invocation.

Reason: the MuJoCo 3.12.0 XML compiler path can call `mj_step` internally for model validation, including a world-only model. That violates this contract's zero dynamic-call limit. A suitable zero-step model allocation path has not been established. Do not remove the entry guard, link/run the probe, load the XML, or consume a pair-call budget under this contract until the model-preparation path and budget are revised and reviewed.

Static checks only: Python Ruff with project config passed; C `cc -std=c11 -Wall -Wextra -Werror -fsyntax-only` against installed typed headers and generated header passed. No executable was linked. No MuJoCo import, model/data construction, pair-kernel call, full archive scan, or new physics call occurred. All new call counters remain zero; original score and qualification/RL gates remain closed.

Input SHA256: fixture `91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55`; kernel contract `0a442f05ba09693d9e97bdc171cee05fef9e1c835225a6bd414209c4c9e374b9`; installed library `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`.
