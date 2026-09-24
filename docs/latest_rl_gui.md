# Latest RL D1 window

**Status: both bounded 07 validation profiles passed independent saved-record readback.** The visible box GUI used the saved final policy; the headless plane profile used zero policy residual with the same baseline controller. Each completed 1,200 controls, 6,000 normal native steps, and three compiler calls, with no retry or new training. The GUI recorded 1,202 rendered frames, three screenshots, and positive wheel-box load. Its selected 0.25 m/s interval averaged 0.2556 m/s across the switched-command box run; this is a descriptive reading, not a fixed-speed qualification score or evidence of independent RL benefit. The earlier 06 results are separate fixed-command evidence.

On this machine, the installed short command opens a new window; `--check` verifies inputs without starting physics:

```bash
rtk proxy d1-rl --check
rtk proxy d1-rl
```

From the repository root, the equivalent source entry is:

```bash
rtk proxy python3 -B scripts/play_d1_latest_rl.py --check
rtk proxy python3 -B scripts/play_d1_latest_rl.py
```

The window starts with the **saved final RL policy**, box terrain, and speed gear 1 (0.20 m/s). Startup-only alternatives are `--actor zero` and `--terrain plane`; for example, `rtk proxy d1-rl --actor zero --terrain plane`. Zero turns off the policy's leg residual while retaining the 06 body-speed and original drive controllers; it does not disable the motors. The real checkpoint and isolated engine library come from the published repository result packages. This is the original 15 mm box or plane model with an oracle-assisted state provider and the fixed patched engine, not a physical-robot trial. Gear 2 selects a 0.25 m/s target. The 07 switched-command speed readings are descriptive; they are not the old fixed-speed qualification score.

Hold **W** to request the selected speed. **1** selects 0.20 m/s; **2** selects 0.25 m/s. Releasing W or pressing **Space/X** requests stop; after Space/X, release W before driving again. **R** makes one simulation reset between control intervals, with the same model and remaining global budget. **Esc** or closing the window saves the session and exits. **C**, mouse, and wheel affect the camera only. Steering, reversing, and jumping are unsupported.

One manual invocation reserves at most two 12-second, 1200-control segments: 2400 controls and 12000 normal native steps globally, plus at most three cold compiler calls. It permits one R reset; an early reset discards the unused portion of the first segment. At a segment boundary the window pauses for R, and the second segment or exhausted budget ends the run. The parent also enforces a 900-second wall-clock limit and never automatically retries. Rendering or a slow CPU can make wall-clock playback slower than simulated time; the display is not a real-time guarantee.

Each launch creates a unique directory under `results/d1_latest_rl_sessions/` with its session reservation, worker output, and launcher receipt. The launcher checks package and source hashes before starting and again after the child exits. A failed or interrupted run keeps its partial evidence in that directory. Only the worker's actual run receipts can establish physical counts and acceptance; the launcher does not infer them.

For the fixed acceptance rules and source identities, see [the frozen 07 contract](latest_rl_gui_contract_07.md). The earlier [05 RL package](../results/d1_rolling_residual_speed_20260924/README.md) and [06 body-speed package](../results/d1_body_speed_feedback_20260924/README.md) document their separate evidence. No independent RL benefit, broad terrain reliability, or hardware behavior is claimed here.
