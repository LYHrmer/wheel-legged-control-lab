#ifndef EPA01_ENGINE_INTERPOSE_H_
#define EPA01_ENGINE_INTERPOSE_H_

#include <stddef.h>
#include <stdint.h>

#include <mujoco/mujoco.h>

#if defined(__GNUC__)
#define EPA01_PUBLIC __attribute__((visibility("default")))
#else
#define EPA01_PUBLIC
#endif

typedef struct {
  uint64_t construction_attempts;
  uint64_t construction_returns;
  uint64_t control_attempts;
  uint64_t control_returns;
  uint64_t ccd_attempts;
  uint64_t ccd_returns;
  uint64_t violations;
  uint64_t construction_limit;
  uint64_t control_limit;
  uintptr_t first_ccd_caller;
  uintptr_t first_construction_step_caller;
  uintptr_t first_control_step_caller;
  uintptr_t original_step;
  uintptr_t target_model;
  uintptr_t target_data;
  int armed;
  int phase;
} epa01_engine_state;

// Phase values: 0 closed, 1 construction, 2 control.
// arm succeeds exactly once; the launcher binds limits to its frozen manifest.
EPA01_PUBLIC int epa01_arm(unsigned construction_limit, unsigned control_limit);
EPA01_PUBLIC int epa01_set_phase(int phase);
EPA01_PUBLIC int epa01_set_target(const mjModel* model, const mjData* data);
EPA01_PUBLIC int epa01_get_state(epa01_engine_state* state, size_t size);
EPA01_PUBLIC uintptr_t epa01_original_step_address(void);

#endif  // EPA01_ENGINE_INTERPOSE_H_
