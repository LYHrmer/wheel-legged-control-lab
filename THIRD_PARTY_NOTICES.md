# Third-party notices

## D1 visual and dynamics assets

The files under `src/wheel_legged_control/d1/assets/urdf/` and
`src/wheel_legged_control/d1/assets/meshes/` are derived from the D1 resources
in [Rangens/WMP-D1-loco](https://github.com/Rangens/WMP-D1-loco), commit
`540e98d0a0c2212bc74908b98088b870a79e2f53`.

That upstream repository distributes these resources under the Apache License
2.0. A copy is retained at
`src/wheel_legged_control/d1/assets/LICENSE-WMP-D1-loco.txt`.

Changes made in this repository:

- `robot.urdf` was renamed from the upstream path and given an explicit MuJoCo
  `discardvisual="false"` compiler extension so the public STL visual meshes
  are retained at runtime.
- Floating-base dynamics, actuators, contact parameters, the ground plane, and
  controllers are added by original Python code in this repository rather than
  copied from another simulator stack.
- The STL mesh files are unmodified.

“D1” and any product appearance or trademarks belong to their respective
owners. This repository is an independent learning project and is not an
official Direct Drive Technology release.
