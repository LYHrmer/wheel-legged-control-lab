// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Arithmetic copied unchanged from fixed MuJoCo 3.12.0
// engine_collision_convex.c:201-203 and 219-232. Names are local to this DSO.
#include "engine/engine_collision_convex.h"
#include "engine/engine_inline.h"

void epa01_engine_pointSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  (void)dir;
  mji_copy3(res, obj->pos);
}

void epa01_engine_lineSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  mjtNum length = obj->size[1];

  mjtNum dot = mat[2]*dir[0] + mat[5]*dir[1] + mat[8]*dir[2];
  mjtNum scl = dot >= 0 ? length : -length;

  res[0] = mat[2]*scl + pos[0];
  res[1] = mat[5]*scl + pos[1];
  res[2] = mat[8]*scl + pos[2];
}
