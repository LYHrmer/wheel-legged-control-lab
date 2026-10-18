"""URDF forward kinematics only: no engine, model compile or native stepping."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'src/wheel_legged_control/d1/assets/urdf/robot.urdf'
tree = ET.parse(PATH).getroot()
joints = tree.findall('joint')
parents = {j.find('child').get('link'): j for j in joints}
reflection = np.diag([1., -1., 1.])
legs = ('FL', 'FR', 'RL', 'RR')


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


def fk(q):
    cache, axes, positions = {}, {}, {}
    def pose(link):
        if link in cache:
            return cache[link]
        if link not in parents:
            return np.eye(4)
        joint = parents[link]
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        roll, pitch, yaw = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        local = np.eye(4)
        local[:3, :3] = rotation([0,0,1], yaw) @ rotation([0,1,0], pitch) @ rotation([1,0,0], roll)
        local[:3, 3] = xyz
        before = pose(joint.find('parent').get('link')) @ local
        name = joint.get('name')
        positions[name] = before[:3, 3]
        if joint.get('type') != 'fixed':
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            axes[name] = before[:3, :3] @ axis
            rot = np.eye(4)
            rot[:3, :3] = rotation(axis, q.get(name, 0.))
            before = before @ rot
        cache[link] = before
        return before
    for joint in joints:
        pose(joint.find('child').get('link'))
    return positions, axes


rng = np.random.default_rng(434)
max_position, max_axis = 0., 0.
rows = []
for sample in range(65):
    q = {}
    for leg in legs:
        angles = [0., .8, -1.5, 0.] if sample == 0 else [
            rng.uniform(-.4, .4), rng.uniform(.3, 1.2), rng.uniform(-2., -1.1), rng.uniform(-2,2)]
        for kind, angle in zip(('hip', 'thigh', 'calf', 'foot'), angles):
            q[f'{leg}_{kind}_joint'] = angle
    mirrored = {}
    for src, dst in zip(legs, ('FR','FL','RR','RL')):
        for kind in ('hip','thigh','calf','foot'):
            mirrored[f'{dst}_{kind}_joint'] = q[f'{src}_{kind}_joint']*(-1 if kind == 'hip' else 1)
    p, a = fk(q)
    pm, am = fk(mirrored)
    pe, ae = [], []
    for src, dst in zip(legs, ('FR','FL','RR','RL')):
        pe.append(float(np.linalg.norm(pm[f'{dst}_foot_joint'] - reflection @ p[f'{src}_foot_joint'])))
        for kind in ('hip','thigh','calf','foot'):
            sign = -1 if kind == 'hip' else 1
            # Rotation axis is an axial vector: det(S)*S*a, with q sign accounted for.
            ae.append(float(np.linalg.norm(sign*am[f'{dst}_{kind}_joint'] + reflection @ a[f'{src}_{kind}_joint'])))
    max_position = max(max_position, *pe)
    max_axis = max(max_axis, *ae)
    rows.append({'sample': sample, 'max_wheel_center_mirror_error_m': max(pe),
                 'max_generalized_axis_mirror_error': max(ae)})
result = {'urdf_sha256': hashlib.sha256(PATH.read_bytes()).hexdigest(),
          'configurations': 65, 'physics_steps': 0, 'environment_constructions': 0,
          'max_wheel_center_mirror_error_m': max_position,
          'max_generalized_axis_mirror_error': max_axis,
          'scope': 'geometry and joint sign audit; does not establish inertial/contact symmetry',
          'rows': rows}
output = ROOT/'reports/b22_symmetry43_geometry_20261018.json'
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
