"""Pre-call ceilings and actual returns for B22 and the separate lateral learner."""
from __future__ import annotations

from functools import wraps

from runtime_support22 import ModelCounter


class ModelLedger31(ModelCounter):
    def _patch(self, owner, name, key, row_argument=None, class_method=False):
        super()._patch(owner, name, key, row_argument, class_method)
        if key not in ('torch_load', 'backward'):
            return
        counted = getattr(owner, name)

        @wraps(counted)
        def guarded(*args, **kwargs):
            allowed = (('probe', 'lateral_final', 'lateral_load')
                       if key == 'torch_load' else ('lateral_update',))
            if self.phase not in allowed:
                raise RuntimeError('C31 forbidden phase for '+key+': '+self.phase)
            phases = self.counts.get(key, {}).get('phases', {})
            cap = 3 if self.phase == 'probe' else 1 if key == 'torch_load' else 64
            if phases.get(self.phase, {}).get('attempted', 0) >= cap:
                raise RuntimeError('C31 phase model ceiling exhausted')
            return counted(*args, **kwargs)

        setattr(owner, name, guarded)


class LateralLedger31:
    """Hooks count executed calls, including failed forwards, before work starts."""
    def __init__(self, *, training, model_counter):
        self.training = bool(training)
        self.model_counter = model_counter
        self.phase = 'construct'
        self.limits = dict(construct=2 if training else 1,
                           actor_rows=2848 if training else 40,
                           value_rows=800 if training else 16,
                           optimizer_step=64 if training else 0,
                           save=1 if training else 0)
        self.counts = {}
        self.phases = {}
        self.policies = []
        self.hooks = []
        self.restore = []

    def before(self, key, amount=1):
        if type(amount) is not int or amount < 1:
            raise RuntimeError('C31 model amount must be a positive integer')
        row = self.counts.setdefault(key, dict(attempted=0, returned=0))
        if row['attempted']+amount > self.limits[key]:
            raise RuntimeError('C31 lateral model ceiling exhausted: '+key)
        row['attempted'] += amount
        stage = self.phases.setdefault(self.phase, {}).setdefault(
            key, dict(attempted=0, returned=0))
        stage['attempted'] += amount

    def after(self, key, amount=1):
        self.counts[key]['returned'] += amount
        self.phases[self.phase][key]['returned'] += amount

    def __enter__(self):
        import torch
        from learning31 import LinearEventPolicy31
        original_init = LinearEventPolicy31.__init__
        original_step = torch.optim.Adam.step
        original_save = torch.save
        ledger = self

        def initialize(policy):
            ledger.before('construct')
            original_init(policy)
            ledger.policies.append(policy)
            for head_name in ('actor', 'value'):
                head = getattr(policy, head_name)
                key = head_name+'_rows'

                def pre(module, args, *, key=key):
                    value = args[0]
                    if (value.ndim != 2 or value.shape[1] != 54
                            or str(value.dtype) != 'torch.float64'
                            or str(value.device) != 'cpu'):
                        raise RuntimeError('C31 lateral actual forward shape/type differs')
                    n = int(value.shape[0])
                    cap = 32 if ledger.phase == 'lateral_update' else 16 if ledger.phase in (
                        'lateral_final', 'lateral_load') else 1
                    if n > cap:
                        raise RuntimeError('C31 actual lateral batch exceeds phase ceiling')
                    ledger.before(key, n)

                def post(module, args, output, *, key=key):
                    if not torch.isfinite(output).all():
                        raise RuntimeError('C31 lateral nonfinite forward')
                    ledger.after(key, int(args[0].shape[0]))

                ledger.hooks.extend((head.register_forward_pre_hook(pre),
                                     head.register_forward_hook(post)))
            ledger.after('construct')

        @wraps(original_step)
        def step(optimizer, *args, **kwargs):
            if ledger.phase != 'lateral_update' or not ledger.policies:
                raise RuntimeError('C31 optimizer outside lateral update')
            expected = {id(p) for p in ledger.policies[0].parameters()}
            actual = {id(p) for group in optimizer.param_groups for p in group['params']}
            if actual != expected or len(optimizer.param_groups) != 2:
                raise RuntimeError('C31 optimizer does not own the unique training parameters')
            ledger.before('optimizer_step')
            result = original_step(optimizer, *args, **kwargs)
            ledger.after('optimizer_step')
            return result

        @wraps(original_save)
        def save(*args, **kwargs):
            if ledger.phase != 'lateral_final':
                raise RuntimeError('C31 save outside final checkpoint')
            ledger.before('save')
            result = original_save(*args, **kwargs)
            ledger.after('save')
            return result

        for owner, name, replacement in ((LinearEventPolicy31, '__init__', initialize),
                                          (torch.optim.Adam, 'step', step),
                                          (torch, 'save', save)):
            self.restore.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)
        return self

    def set_phase(self, phase):
        self.phase = self.model_counter.phase = phase

    def __exit__(self, *_):
        for hook in self.hooks:
            hook.remove()
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)

    def report(self):
        return dict(schema='d1-c31-actual-lateral-model-ledger-v1', limits=self.limits,
                    counts=self.counts, phases=self.phases,
                    actual_parameter_objects=len(self.policies))


def parameters31(policy):
    return {name: value.detach().cpu().tolist()
            for name, value in policy.state_dict().items()}
