# D1 handoff — 2026-09-29

The project goal remains a useful, visibly drivable wheel-legged RL portfolio
demo with faster forward motion, reliable obstacle crossing and a responsive
GUI. The fixed patched simulation engine, 92-geom course, oracle state
provider, 99D observation and 16D leg/wheel residual are the present research
environment. Nothing here establishes real-hardware behavior or broad
statistical reliability.

The earlier 10 terminal probe closed with independent readback. It showed the
last real task failure can return a finite post-state observation without
refunding native steps; it did not change the original failure's outcome.
The 11-R world-upright zero-residual ramp baseline passed 1,800 controls and
9,000 native steps, including all four wheel collision shapes beyond the
compiled ramp. The 11-M 1,024-step synthetic model preflight passed its pure
interface and strict reload checks; it is not a physical RL result.

The one new 11-S run completed 65,536 training controls (64 rollout/train
calls, 256 epochs, 1,024 optimizer steps) and uniquely saved/reloaded a final
checkpoint. Its host watchdog expired during the tenth heldout case. Nine
cases have complete worker records; the rough final-policy case is partial
(1,024 controls attempted, 1,023 returned; all last five native steps
returned), and both ramp actors never started. C recorded 404,800 returned
normal native steps with no violation; host source hashes remained unchanged,
no child orphan remained and the reserved budget is closed. An interrupted
gzip write followed by a sealing `FileExistsError` means the tenth block is
not usable as a complete heldout case. The
[independent saved-data readback](../results/d1_rl16_progress_20260929/evidence/rl11/partial_readback_20260929_05.json)
verified training and checkpoint identity, nine complete native/contact
chains, and four bitwise-identical zero/policy initial-state pairs without
replaying physics or loading the model. All nine complete cases passed their
task gates. In the single-seed scripted flat 1.6 m/s case, zero and policy
COM mean speeds were 1.610807 and 1.632405 m/s; the policy held above
1.5 m/s for 400/400 ticks and passed the stop gate at 3.37 s and 2.642 m
maximum forward excursion. This establishes that flat scripted speed case,
not 1.6 m/s across terrain or in the GUI. Across the four complete pairs,
policy drive tracking SSE rose by 53.054%, 132.413%, 9.494% and 15.578%,
respectively; the normalized torque-squared proxy also rose in every pair.
These comparisons do not establish RL gain. Rough policy and both ramp
actors remain unknown, so neither 12-case qualification nor overall
matched-pair contribution is complete.

The new compact progress package is
`results/d1_rl16_progress_20260929/`. It carries the unique checkpoint with
its **unqualified-for-GUI** status, exact source and small receipts. Gigabyte
raw records remain local and are not presented as a GitHub release asset.
The raw Claude 09 GUI session reported a different assistant model identity
from its requested Opus setting; count those files only as unverified source
candidates, not an Opus contribution. GUI12 has unfinished candidate source
and no accepted execution. The Q compact GUI's actual performance gate failed;
its profiles do not prove smooth real-time operation. A/D lateral motion and
Space jump are still absent from a qualified runnable flow.

The next bounded work is detailed in the [2026-09-29 main plan](main_plan_20260929.md)
and [engineering follow-up](rl16_engineering_followup_20260929.md). Preserve
the 11-S run, including its damaged block and host/worker receipts. First
make block publication atomic and interruption safe, then measure real
overhead under a separate limited contract. Inspect the nine complete cases
before deciding whether any missing terrain evaluation can change a decision.
Any such evaluation needs a fresh explicit budget and must disclose
cross-process model/data identity; it cannot be spliced into the original
single-process qualification. Do not credit RL benefit by changing only the
report.

For GitHub, publish the compact package, this handoff and the finalized
optimization plan after checking their source hashes. Record the actual
commit, remote main, CI result and final publication receipt after those
steps; this file does not predeclare them. Keep user private driving results,
Xauthority/cookies and raw Claude logs out of the publication.
