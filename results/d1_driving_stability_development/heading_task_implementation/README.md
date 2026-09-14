# Heading task implementation and provenance

GPT-6-astra / ultra specified the observable heading task. Actual Claude Opus supplied the 635-line original environment and compatibility loader (`d1_heading_tracking_env.original.py`); root integrated it and fixed immutable task identity after an independent reproduction. `opus_contribution.json` records the actual provider, session, original SHA and local changes. No model-performance claim follows from this implementation.

Validation includes 26 environment checks, a physical parity check for the independent evaluator, 15 curriculum/runner tests and real 32/256-transition PPO smoke runs. See [smoke evidence](training_smoke/README.md); the first failed validation run is retained. The independent reviews record checks of reference timing, noisy measurements versus reward truth, cached terminal observations, checkpoint identities and common-prefix metrics.

`heading_training_protocol.json` was frozen before the first full training seed: 49001/49002/49003, 65,536 steps each; checkpoints at 16,384 and 65,536 **after** completed PPO updates; a fixed 16,384-transition curriculum denominator; evaluation on the already opened development road. This training only covers straight commands with zero user yaw rate. Full stopping, turning, GUI and obstacle acceptance remain outstanding.

Core: [environment](../../../scripts/d1_heading_tracking_env.py), [training](../../../scripts/run_d1_heading_study.py), [evaluation](../../../scripts/evaluate_d1_heading_study.py).
