# First jump PPO failure diagnosis

The formal 131072-control-step run and ten-episode evaluation are valid records of a failed policy. This package adds **0 integration steps and 0 training steps**. It neither reruns nor supersedes those results.

## What the saved data establishes

- Training produced 16 positive jump-event transitions among 21072 requested-window transitions (0.07593%). Sixteen of 175 complete jump episodes received a flight event; none reached full clearance progress or received landing credit. Counts are the original control-quantized training accounting, not a new physical certification. There were 218 complete 600-step episodes and one 272-step partial episode. PPO did update: logged policy standard deviation changed from about 0.13534 to 0.12328.
- The final policy received no jump bonus in any of the four independent jump evaluations. Its genuine request-window native measurements are below. These exclude reset settling and use executions [s,s+120), with 2 ms returned geometry and corresponding recorded contact loads. Net gap subtracts the original 1 mm margin; the goal remains 20 mm net gap and at least 20 ms certified flight.

| Case | Zero net gap peak, mm | Final net gap peak, mm | Zero longest unloaded run, ms | Final longest unloaded run, ms |
|---|---:|---:|---:|---:|
| late_a, s=175 | 0.49184 | 0.46775 | 12 | 10 |
| late_b, s=225 | 0.49120 | 0.27761 | 12 | 8 |
| lower friction | 0.56122 | 0.57642 | 14 | 12 |
| higher friction | 0.58149 | 0.07067 | 12 | 4 |

- The final requested actions use all eight residual channels. Only 19.65–22.62% of squared leg-action norm lies along (1,1,1,1); the remainder changes relative leg shape. This is an orthogonal projection of action coordinates, not a share of energy or a causal attribution. Wheel residuals are also nonzero. There was no torque protection or extension-table clipping in these four final request windows, but actual **leg** joint targets hit the original measured-position/rated-speed increment bound in 32–34 of 120 intervals. Unsaturated torque does not prove sufficient useful takeoff authority.
- The final no-request hold drifts 0.156127 m versus the zero baseline's 0.001089 m. Its return falls from 11.930742 to 8.689095. Some jump-case late velocity peaks improve, while clearance and planar displacement do not improve consistently. These are mixed stability effects, not a successful jump skill.
- Upward robot-COM motion is real: in late_a the peak requested COM vertical speed is about 0.647 m/s for zero and 0.619 m/s for final. It still fails to sustain unloaded wheel clearance. Body rise, COM ascent, wheel clearance, and actual contact-free duration must remain separate measurements.

## Interpretation and limits

Jump credit is sparse, but it is not absent; useful shaping and repeated success did not emerge in this run. The saved data is consistent with incoherent independent leg/wheel exploration and a stability-seeking policy, but does not establish either as the unique cause. A scalar common-leg action is a falsifiable task-specific exploration hypothesis. It may remove necessary asymmetric landing corrections and it may still fail to achieve 20 mm. Neither controller torque headroom nor action projection proves reachability.

Training's compact all-16 joint target-rate flag includes wheel slots whose position targets are copied from measured wheel angles. It must not be interpreted as all-leg slew saturation. The evaluator analysis explicitly excludes joint slots 3,7,11,15. No endpoint velocity was multiplied by a differently phased native force to manufacture work or power.

## Reproduction

`analyze.py` reads JSON/JSONL/gzip/NPZ records with NumPy only. It never imports MuJoCo, an environment, controller, or policy. Run from any directory with a **new** output path, preserving this report:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/rl_jump_failure_plan_01/analyze.py --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --output /tmp/d1_jump_failure_reproduced.json
```

The report includes input hashes. `next_contract.md` is the sole next proposal: zero-integration qualification of a one-dimensional stationary-hop action embedding. No second training budget is started or authorized by this report. Old three 65k studies, 24 G1 runs, the completed 131072-step run, and these ten evaluations remain untouched and are not rerun.
