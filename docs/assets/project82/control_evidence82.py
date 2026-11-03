"""Read historical sensor/PI CSVs independently; never import simulator code."""
import argparse
import csv
import hashlib
import json
import math
import resource
import signal
import time
from pathlib import Path


def mean(v):
    return math.fsum(v) / len(v)


def rms(v):
    return math.sqrt(mean([x*x for x in v]))


def run(root):
    inputs, checks, gaps = {}, 0, {}

    def bind(path):
        path = path.resolve()
        data = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def js(path):
        return json.loads(bind(path))

    def rows(path):
        return list(csv.DictReader(bind(path).decode().splitlines()))

    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            raise ValueError(label)

    def close(a, b, label):
        gap = abs(a-b)
        gaps[label] = max(gaps.get(label, 0), gap)
        check(math.isfinite(a) and math.isfinite(b) and gap <= 1e-11, label)

    sensor_dir = root/'results/d1_sensor_estimation/development_multiseed'
    summary = js(sensor_dir/'summary.json')
    sensor = []
    check(len(summary['cases']) == 144, 'all 144 sensor cases')
    for case in summary['cases']:
        tag = f"{case['mode']}_{case['terrain']}_{case['noise']}_delay{case['delay_steps']}_seed{case['seed']}"
        rr = rows(sensor_dir/(tag+'.csv'))
        check(len(rr) == case['steps'], 'sensor prefix length')
        pairs = {
            'pitch_rmse_rad': ('estimate_pitch_rad', 'truth_pitch_rad'),
            'velocity_estimation_rmse_mps': ('estimate_velocity_body_x_mps', 'truth_velocity_body_x_mps'),
            'clearance_estimation_rmse_m': ('estimate_clearance_m', 'truth_clearance_m'),
            'ground_pitch_rmse_rad': ('estimate_ground_pitch_rad', 'truth_ground_pitch_rad'),
            'tracking_rmse_mps': ('truth_velocity_body_x_mps', 'command_velocity_mps'),
        }
        reconstructed = {}
        for label, (a, b) in pairs.items():
            reconstructed[label] = rms([float(r[a])-float(r[b]) for r in rr])
            close(reconstructed[label], case[label], 'sensor_'+label)
        ages = []
        for i, r in enumerate(rr):
            age = float(r['time_s'])-float(r['measurement_time_s'])
            close(age, float(r['measurement_age_s']), 'sensor_age_field')
            close(age, min((i+1)*.01, case['delay_steps']*.01), 'sensor_queue_age')
            ages.append(age)
        sensor.append({k: case[k] for k in ('mode','terrain','noise','seed','delay_steps','steps','termination_reason')} | {
            'metrics_reconstructed': reconstructed, 'maximum_age_s': max(ages),
            'support_plane_unobservable_fraction': mean([float(r['support_plane_observable']) == 0 for r in rr]),
            'accelerometer_unused_fraction': mean([float(r['accelerometer_used']) == 0 for r in rr]),
            'contacts_less_than_3_fraction': mean([float(r['contact_switches']) < 3 for r in rr]),
        })
    sensor_counts = []
    for mode in ('offline','closed_loop'):
        for delay in (0,2):
            ss = [c for c in sensor if c['mode']==mode and c['delay_steps']==delay]
            sensor_counts.append({'mode':mode,'delay_steps':delay,'episodes':len(ss),
                'full_400':sum(c['steps']==400 and c['termination_reason']=='time_limit' for c in ss),
                'termination_reasons':{reason:sum(c['termination_reason']==reason for c in ss) for reason in sorted({c['termination_reason'] for c in ss})},
                'steps':sum(c['steps'] for c in ss)})

    pi_dir = root/'results/pi49_preflight_20261022_01'
    pi_summary = js(pi_dir/'summary.json')
    check(pi_summary['status']=='complete' and len(pi_summary['episodes'])==12, 'complete12 PI evidence')
    pi = []
    inertia = .5*.5*.087**2
    fit_info = {}
    for ep in pi_summary['episodes']:
        case = f"{ep['seed']}_{ep['scenario']}_{ep['profile']}_{ep['arm']}"
        rr = rows(pi_dir/case/'trajectory.csv')
        check(len(rr)==ep['steps']==3000, 'PI complete3000')
        fit = js(root/f"results/wheel_control/{ep['scenario']}/fit.json")
        fit_info[ep['scenario']] = {'parameters':fit['parameters'],'fit_status':fit['fit_status']}
        p, scale = fit['parameters'], 1. if ep['arm']=='baseline' else .75
        counts = {'inertial_ff_over_2Nm':0,'steady_ff_over_2Nm':0,'total_unscaled_ff_over_2Nm':0,
                  'oracle_tracking_torque_over_2Nm':0,'saturated':0,'frozen':0,
                  'saturated_while_inertial_ff_le_2Nm':0}
        reference_required = []
        for i, r in enumerate(rr):
            v, acc = float(r['reference_velocity_rad_s']), float(r['reference_acceleration_rad_s2'])
            measured = float(r['measured_velocity_rad_s'])
            actual = float(r['actual_velocity_rad_s'])
            error = v-measured
            inertial = (inertia+p['armature'])*acc
            steady = p['damping']*v+p['coulomb_friction']*math.tanh(v/.05)
            expected_ff = scale*(inertial+steady)
            close(expected_ff, float(r['feedforward_torque_nm']), 'PI_feedforward')
            close(error, float(r['error_rad_s']), 'PI_measurement_error')
            close(v-actual, float(r['tracking_error_rad_s']), 'PI_truth_error')
            close(.12*error, float(r['proportional_torque_nm']), 'PI_P')
            old_i = float(rr[i-1]['integral_torque_nm']) if i else 0.
            inc = .6*error*.002
            proposed = .12*error+old_i+inc+expected_ff
            frozen = (proposed>2 and inc>0) or (proposed < -2 and inc<0)
            check(frozen == (r['integration_frozen']=='True'), 'PI antiwindup branch')
            accepted_i = old_i if frozen else old_i+inc
            close(accepted_i,float(r['integral_torque_nm']),'PI integral')
            requested = .12*error+accepted_i+expected_ff
            close(requested,float(r['requested_torque_nm']),'PI requested')
            close(max(-2,min(2,requested)),float(r['command_torque_nm']),'PI command clip')
            close(float(rr[i-2]['command_torque_nm']) if i>=2 else 0.,float(r['applied_torque_nm']),'PI true two-tap delay')
            saturated = abs(requested)>2
            check(saturated == (r['saturated']=='True'),'PI saturation flag')
            # Algebraic torque needed to follow the reference exactly on the known
            # synthetic plant, ignoring command delay. Diagnostic only, not a new rollout.
            friction = .043 + (.08*math.exp(-(v/.3)**2) if ep['scenario']=='mismatch' else 0)
            required = (inertia+.017)*acc+.105*v+friction*math.tanh(v/.05)
            reference_required.append(abs(required))
            counts['inertial_ff_over_2Nm'] += abs(inertial)>2
            counts['steady_ff_over_2Nm'] += abs(steady)>2
            counts['total_unscaled_ff_over_2Nm'] += abs(inertial+steady)>2
            counts['oracle_tracking_torque_over_2Nm'] += abs(required)>2
            counts['saturated'] += saturated
            counts['frozen'] += frozen
            counts['saturated_while_inertial_ff_le_2Nm'] += saturated and abs(inertial)<=2
        rmse = rms([float(r['tracking_error_rad_s']) for r in rr])
        close(rmse, ep['velocity_rmse_rad_s'], 'PI tracking RMSE')
        close(counts['saturated']/len(rr), ep['saturation_fraction'], 'PI saturation fraction')
        close(counts['frozen']/len(rr), ep['integration_frozen_fraction'], 'PI frozen fraction')
        pi.append({k:ep[k] for k in ('seed','scenario','profile','arm')} | {
            'rows':len(rr),'velocity_rmse_rad_s':rmse,'counts':counts,
            'fractions':{k:v/len(rr) for k,v in counts.items()},
            'known_synthetic_plant_reference_required_peak_Nm':max(reference_required)})
    source_status = []
    for path, expected in summary['source_sha256'].items():
        p = Path(path) if Path(path).is_absolute() else root/path
        actual = hashlib.sha256(bind(p)).hexdigest()
        source_status.append({'path':str(p),'historical_sha256':expected,'current_sha256':actual,'matches':actual==expected})
    for name in ('src/wheel_legged_control/wheel_control.py','src/wheel_legged_control/actuator_bench.py','scripts/control_candidates49.py'):
        bind(root/name)
    for name, digest in inputs.items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, 'input changed')
    return {'status':'COMPLETE','schema':'control-evidence82-readback-v1','checks':checks,
            'inputs_sha256':inputs,'maximum_reconstruction_gaps':gaps,
            'sensor_historical_source_status':source_status,'sensor_counts':sensor_counts,'sensor_cases':sensor,
            'pi_fit_info':fit_info,'pi_episodes':pi,'pi_original_decision':pi_summary['decision'],
            'new_physics_steps':0,'new_model_calls':0,'new_parameter_updates':0,
            'interpretation_limits':[
                'Historical completed early-terminal sensor episodes are valid failure evidence, not partial runs.',
                'Current source hashes are compared to historical metadata; mismatch is reported, not silently relabelled.',
                'Age consistency and conditional guard associations do not establish the causal reason for falls.',
                'No per-wheel odometry residual is available in these CSVs. A robust aggregation replacement is not validated.',
                'Oracle reference torque is derived from known synthetic parameters and does not identify a real motor.',
                'Exceeding2Nm rules out exact tracking at those instants under that model; it gives no lower bound on full-task RMSE.',
                'No new controller is simulated. Saturation decomposition alone cannot certify an alternative feedforward.',
                'The current PI already conditionally freezes its integral. Adding generic antiwindup is not a missing-feature fix.']}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    resource.setrlimit(resource.RLIMIT_CPU,(29,30))
    def stop(*_): raise TimeoutError('readback budget exhausted')
    signal.signal(signal.SIGXCPU,stop); signal.signal(signal.SIGALRM,stop); signal.alarm(60)
    start=time.process_time()
    try:
        result=run(Path('/home/lyh/wheel-legged-control-lab'))
        result['cpu_seconds']=time.process_time()-start
        result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if len(payload.encode())>2<<20: raise ValueError('output exceeds2MiB')
    except BaseException as exc:
        result={'status':'INCOMPLETE','error':str(exc),'cpu_seconds':time.process_time()-start,
                'new_physics_steps':0,'new_model_calls':0,'new_parameter_updates':0}
        payload=json.dumps(result,allow_nan=False)+'\n'
    with a.output.open('x') as out: out.write(payload)
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('status','checks','cpu_seconds','error')}))
    return int(result['status']!='COMPLETE')


if __name__=='__main__': raise SystemExit(main())
