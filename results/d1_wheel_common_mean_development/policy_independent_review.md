# Wheel common mean policy independent review

Date: 2026-09-14. Scope: `scripts/d1_wheel_common_mean_policy.py` and its new test file in `/home/lyh/wheel-legged-control-lab`; local installed Stable-Baselines3 2.9.0. No project source was edited by this reviewer, and no frozen 77-study source was touched.

**Disposition: no blocker found for the frozen float32, c=0.05 / None experiment.** The mean transformation uses the ordinary diagonal Gaussian consistently in rollout, prediction and PPO updates. Two low-priority API boundary limitations are reproduced below; neither affects the selected experiment configuration.

## Independently verified behavior

- Inspected the installed SB3 `forward`, `get_distribution`, `_predict`, `evaluate_actions`, and `collect_rollouts`. All relevant policy paths reach the overridden `_get_action_dist_from_latent`. `collect_rollouts` stores the original sampled action with its corresponding Gaussian log probability, while passing the clipped action to the environment. The new policy does not change this behavior or apply a tanh Jacobian to samples.
- Ran a synthetic 82-observation / 8-action Gym environment through PPO with `n_steps=8`, `batch_size=8`, `n_epochs=2`, `total_timesteps=16`. At each rollout end, independently recomputed the normal log probabilities and compared them with both the rollout buffer and `evaluate_actions`; all checks passed. Environment actions were finite and within [-1,1], and all parameters remained finite after both updates.
- Ran that same training procedure with stock `ActorCriticPolicy` and the new policy with `wheel_common_mean_limit=None`, each initialized with seed 81. All final state-dict tensors, rollout actions and stored log probabilities were bit-for-bit equal. Stochastic and deterministic `predict` were also bit-for-bit equal under matched RNG seeds.
- Tested positional constructor arguments enabling gSDE, squash, and both. None could instantiate an unsupported distribution. Exact distribution type checking catches positional gSDE after base construction.
- Set actor wheel biases to `[3,-1,-1,-1]`, making the pre-clipping common mean zero. Deterministic `predict` returned clipped wheel actions with common mean **-0.5**, as the documentation warns. The bound must remain described as a bound on the pre-sampling, pre-clipping distribution mean.
- Reviewed save/load implementation and the root-authored policy/PPO roundtrip tests for `None`, `0.05`, and `0.125`. The policy constructor serialization includes the limit, and PPO retains the policy class and policy kwargs. Root separately reported all 18 policy tests passing; these existing tests were not redundantly rerun by this reviewer.

Independent runtime command prefix: `rtk proxy env PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B`, working directory the project root. Checks ran as read-only inline scripts.

## Reproducible, non-blocking boundary limitations

1. **Positional squash rejection has a different exception type.** At constructor lines 223–235, keyword guards do not bind `*args` to the base signature. Passing positional `use_sde=False, squash_output=True` reaches SB3's `AssertionError: squash_output=True is only available when using gSDE (use_sde=True)`, whereas the new constructor documents `ValueError`. This is an exception-contract mismatch, not an acceptance bypass. The experiment uses keyword defaults, so no change is needed to proceed.

2. **The public numeric contract is wider than float32 arithmetic supports.** At lines 97–103 and 139, any finite positive Python float passes validation, even when it overflows or underflows in the input tensor's dtype. For float32 wheel means `[0.1,0.1,0.1,0.1]`, `c=1e40` gives NaN wheel outputs and gradients; `c=1e-50` gives zero wheel outputs with non-finite gradients. The default `c=0.05` on the identical input gives finite wheel output `0.0482013784` and finite gradients. These values are outside the frozen parameter choice. If the API is later generalized, constrain/document the supported numeric range or reject non-representable limits; this is not a reason to broaden the present experiment.

## Practical limit of this review

This review validates the policy's probability semantics, selected numeric configuration and software integration. It does not establish that the transformed policy improves physical tracking, nor that executed common wheel actions stay bounded after native clipping. Those claims require the separately defined MuJoCo evaluation and action/clipping diagnostics.

## Integration follow-up

Codex subsequently added dtype-aware rejection of non-normal/overflowed limits at helper and policy construction. The 1e40 and 1e-50 float32 counterexamples now raise ValueError; float64 tensors still produce finite outputs and gradients. Both cases have regression tests. The selected .05/None formula is unchanged.
