"""Independent explicit-reference arithmetic; no simulation or imports of package init."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

root = Path('/home/lyh/wheel-legged-control-lab')
module = root / 'src/wheel_legged_control/reference_feasibility83.py'
spec = importlib.util.spec_from_file_location('feasibility83', module)
loaded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaded)
dt, inertia, damping = .002, .5 * .5 * .087**2 + .017, .105
reference = [18 * math.sin(2 * math.pi * k * dt) for k in range(3001)]
result = loaded.check_reference_torque(reference, dt=dt, total_inertia=inertia,
    damping=damping, coulomb=.043, stribeck=.08, torque_limit=2.)
independent = [inertia*(b-a)/dt + damping*b + (.043+.08*math.exp(-(a/.3)**2))*math.tanh(a/.05)
               for a,b in zip(reference, reference[1:])]
gaps = [abs(row['required_torque_nm']-value) for row,value in zip(result['intervals'],independent,strict=True)]
assert max(gaps) < 1e-12
assert sum(abs(x)>2 for x in independent) == result['violation_count'] == 1512
assert math.isclose(max(map(abs,independent)), 2.90448655515, rel_tol=0, abs_tol=1e-10)
paths = [root / f'results/pi49_preflight_20261022_01/{seed}_matched_stress_baseline/trajectory.csv'
         for seed in (490101,490102,490103)]
archive = []
for path in paths:
    with path.open() as f:
        values = [float(row['reference_velocity_rad_s']) for row in csv.DictReader(f)]
    assert len(values) == 3000
    # Same deterministic analytic reference; numerical evaluation order can differ.
    gap = max(abs(a-b) for a,b in zip(values, reference[:3000], strict=True))
    assert gap < 1e-10
    archive.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'sample_count': len(values), 'max_reference_gap': gap})
result.update(status='PASS', module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
              independent_max_torque_gap=max(gaps), archive_reference_checks=archive,
              physics_steps=0, model_calls=0, parameter_updates=0,
              scope='One fixed deterministic reference, not three independent seed estimates')
with Path('/home/lyh/project83_work_20261104/pi_readback83.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ('status','interval_count','violation_count','violation_fraction','peak_abs_required_torque_nm','necessary_torque_condition')}))
