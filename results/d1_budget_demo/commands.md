# Demonstration commands

Executed from `/home/lyh/wheel-legged-control-lab`. All outputs are new paths under
`/tmp/d1_demo_delivery_2hwlvrso`; the formal study and repository source are read only.
The two 60 s runs execute sequentially with one BLAS/OpenMP thread. Policy inference
also sets Torch to one thread in the existing CLI.

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab/.local-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_d1_locomotion.py --baseline wheel_leg --action-mode independent8 --source sensor --seed 1017 --terrain-json /tmp/d1_demo_delivery_2hwlvrso/terrain.json --command-mode development --seconds 60 --wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4 --leg-feedback-scale 1 --attitude-feedback-scale 0.25 --output /tmp/d1_demo_delivery_2hwlvrso/zero

rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab/.local-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_d1_locomotion.py --baseline wheel_leg --action-mode independent8 --source sensor --seed 1017 --terrain-json /tmp/d1_demo_delivery_2hwlvrso/terrain.json --command-mode development --seconds 60 --wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4 --leg-feedback-scale 1 --attitude-feedback-scale 0.25 --policy /home/lyh/wheel-legged-control-lab/results/d1_budget_study/independent8_seed31000/checkpoints/budget262144/checkpoint.zip --metadata /home/lyh/wheel-legged-control-lab/results/d1_budget_study/independent8_seed31000/checkpoints/budget262144/checkpoint.json --output /tmp/d1_demo_delivery_2hwlvrso/ppo

rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 /tmp/d1_demo_delivery_2hwlvrso/gui_check.py --root /home/lyh/wheel-legged-control-lab --output /tmp/d1_demo_delivery_2hwlvrso/gui_retry
```

The GUI helper logs its exact child/capture commands, owned window PID/ID, planned
and actual event times in `gui_retry/events.json`. It injects X11 events into the real
MuJoCo viewer; this is automated GUI integration validation, not human acceptance.
The capture uses FFmpeg's `-window_id` and never captures the desktop.

The first GUI attempt is retained in `gui/`. Its key/telemetry sequence passed,
but signaling the entire FFmpeg wrapper group interrupted MP4 finalization
(`moov atom not found`). `gui_check_first_attempt.py` preserves that helper version.
The successful retry uses FFmpeg's own fixed duration to end normally. Its window
video fully decodes as 132 frames at 20 fps (6.6 s). The rollout stops via Escape,
so CLI exit 1 and `completed=false` are expected; `stop_reason=keyboard_escape`
distinguishes this requested stop from falling or numerical failure.

`terrain.json` exactly matches the first formal development road, selected before
running either policy. Reset seed 1017 derives measurement seed 1972145375 and
command seed 2119272573, matching the formal development episode. The source CLI
alias `sensor` maps to `imu_encoder_fusion`. `pair_audit.json` checks those values,
the compiled model, initial qpos/qvel, and all 6000 executed commands.

The PPO is the predetermined independent8 training seed 31000 final 262144 sample
checkpoint, not a model selected by holdout score. Both runs complete 60 s; on this
one development case PPO reduces height RMSE but increases velocity/yaw RMSE.
This demonstration does not establish aggregate holdout performance.

Render the saved synchronized states; these commands run no physics or policy:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab/.local-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/render_d1_locomotion.py --run /tmp/d1_demo_delivery_2hwlvrso/zero --output /tmp/d1_demo_delivery_2hwlvrso/zero.mp4 --fps 20

rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab/.local-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/render_d1_locomotion.py --run /tmp/d1_demo_delivery_2hwlvrso/ppo --output /tmp/d1_demo_delivery_2hwlvrso/ppo.mp4 --fps 20

rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 /tmp/d1_demo_delivery_2hwlvrso/compose_pair.py

rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 /tmp/d1_demo_delivery_2hwlvrso/audit_demo.py
```

The first renderer ran with software EGL inside the sandbox. Since `/dev/dri` is
absent there, a separately approved five-frame host-device probe checked hardware
rendering before the PPO render. Timing evidence is in `probe_lp1/timing.json` and
`probe_gpu/timing.json`; geometry, camera, rendering source, and frame selection
remain the same. Composition checks every frame's saved-state index before joining
the videos, and the final audit decodes all delivered MP4 files completely.
