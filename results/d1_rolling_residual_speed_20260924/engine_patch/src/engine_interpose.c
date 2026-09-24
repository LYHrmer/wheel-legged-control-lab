#define _GNU_SOURCE

#include "engine_interpose.h"

#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "engine/engine_collision_gjk.h"

// The generated clean GJK source implements these under private names.
size_t epa01_engine_local_ccdSize(int npolygonmax, int nmeshdegmax, int iterations);
mjtNum epa01_engine_local_ccd(const mjCCDConfig* config, mjCCDStatus* status,
                             mjCCDObj* obj1, mjCCDObj* obj2);

static _Atomic int armed = 0;
static _Atomic int phase = 0;
static _Atomic uintptr_t target_model = 0;
static _Atomic uintptr_t target_data = 0;
static _Atomic uint64_t construction_attempts = 0;
static _Atomic uint64_t construction_returns = 0;
static _Atomic uint64_t control_attempts = 0;
static _Atomic uint64_t control_returns = 0;
static _Atomic uint64_t ccd_attempts = 0;
static _Atomic uint64_t ccd_returns = 0;
static _Atomic uint64_t violations = 0;
static _Atomic uint64_t configured_construction_limit = 0;
static _Atomic uint64_t configured_control_limit = 0;
static _Atomic uintptr_t first_ccd_caller = 0;
static _Atomic uintptr_t first_construction_step_caller = 0;
static _Atomic uintptr_t first_control_step_caller = 0;
static _Atomic uintptr_t original_step = 0;

static void hard_stop(const char* reason) {
  atomic_fetch_add_explicit(&violations, 1, memory_order_relaxed);
  static const char prefix[] = "epa01 engine guard: ";
  (void)write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
  (void)write(STDERR_FILENO, reason, strlen(reason));
  (void)write(STDERR_FILENO, "\n", 1);
  _exit(125);
}

static void remember_first(_Atomic uintptr_t* slot, uintptr_t value) {
  uintptr_t empty = 0;
  atomic_compare_exchange_strong_explicit(slot, &empty, value, memory_order_relaxed,
                                          memory_order_relaxed);
}

static void reserve_step(_Atomic uint64_t* attempts, _Atomic uint64_t* limit,
                         const char* exceeded_reason) {
  uint64_t current = atomic_load_explicit(attempts, memory_order_acquire);
  for (;;) {
    if (current >= atomic_load_explicit(limit, memory_order_acquire)) {
      hard_stop(exceeded_reason);
    }
    if (atomic_compare_exchange_weak_explicit(attempts, &current, current + 1,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) return;
  }
}

static void (*real_step(void))(const mjModel*, mjData*) {
  uintptr_t address = atomic_load_explicit(&original_step, memory_order_acquire);
  if (!address) {
    void* symbol = dlsym(RTLD_NEXT, "mj_step");
    if (!symbol || symbol == (void*)&mj_step) {
      hard_stop("RTLD_NEXT mj_step unavailable");
    }
    address = (uintptr_t)symbol;
    uintptr_t empty = 0;
    atomic_compare_exchange_strong_explicit(&original_step, &empty, address,
                                            memory_order_acq_rel, memory_order_acquire);
    address = atomic_load_explicit(&original_step, memory_order_acquire);
  }
  return (void (*)(const mjModel*, mjData*))address;
}

EPA01_PUBLIC int epa01_arm(unsigned construction_limit, unsigned control_limit) {
  int expected = 0;
  if (!atomic_compare_exchange_strong_explicit(&armed, &expected, 2,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) return -1;
  atomic_store_explicit(&configured_construction_limit, construction_limit, memory_order_release);
  atomic_store_explicit(&configured_control_limit, control_limit, memory_order_release);
  atomic_store_explicit(&armed, 1, memory_order_release);
  return 0;
}

EPA01_PUBLIC int epa01_set_phase(int next_phase) {
  if (next_phase < 0 || next_phase > 2) return -1;
  if (atomic_load_explicit(&armed, memory_order_acquire) != 1 && next_phase != 0) return -1;
  if (next_phase == 2 && (!atomic_load_explicit(&target_model, memory_order_acquire)
                       || !atomic_load_explicit(&target_data, memory_order_acquire))) return -1;
  atomic_store_explicit(&phase, next_phase, memory_order_release);
  return 0;
}

EPA01_PUBLIC int epa01_set_target(const mjModel* model, const mjData* data) {
  if (!!model != !!data) return -1;
  // The frozen control loop reaffirms the same allowed pair each interval.
  // A live control phase cannot switch to a different pair or clear it.
  if (atomic_load_explicit(&phase, memory_order_acquire) == 2
      && ((uintptr_t)model != atomic_load_explicit(&target_model, memory_order_acquire)
          || (uintptr_t)data != atomic_load_explicit(&target_data, memory_order_acquire))) return -1;
  atomic_store_explicit(&target_model, (uintptr_t)model, memory_order_release);
  atomic_store_explicit(&target_data, (uintptr_t)data, memory_order_release);
  return 0;
}

EPA01_PUBLIC uintptr_t epa01_original_step_address(void) {
  return (uintptr_t)real_step();
}

EPA01_PUBLIC int epa01_get_state(epa01_engine_state* state, size_t size) {
  if (!state || size != sizeof(*state)) return -1;
  state->construction_attempts = atomic_load_explicit(&construction_attempts, memory_order_acquire);
  state->construction_returns = atomic_load_explicit(&construction_returns, memory_order_acquire);
  state->control_attempts = atomic_load_explicit(&control_attempts, memory_order_acquire);
  state->control_returns = atomic_load_explicit(&control_returns, memory_order_acquire);
  state->ccd_attempts = atomic_load_explicit(&ccd_attempts, memory_order_acquire);
  state->ccd_returns = atomic_load_explicit(&ccd_returns, memory_order_acquire);
  state->violations = atomic_load_explicit(&violations, memory_order_acquire);
  state->construction_limit = atomic_load_explicit(&configured_construction_limit,
                                                   memory_order_acquire);
  state->control_limit = atomic_load_explicit(&configured_control_limit, memory_order_acquire);
  state->first_ccd_caller = atomic_load_explicit(&first_ccd_caller, memory_order_acquire);
  state->first_construction_step_caller = atomic_load_explicit(
      &first_construction_step_caller, memory_order_acquire);
  state->first_control_step_caller = atomic_load_explicit(
      &first_control_step_caller, memory_order_acquire);
  state->original_step = atomic_load_explicit(&original_step, memory_order_acquire);
  state->target_model = atomic_load_explicit(&target_model, memory_order_acquire);
  state->target_data = atomic_load_explicit(&target_data, memory_order_acquire);
  state->armed = atomic_load_explicit(&armed, memory_order_acquire);
  state->phase = atomic_load_explicit(&phase, memory_order_acquire);
  return 0;
}

// The exact original ABI must be exported. The local kernel receives all
// original configuration, status, and object pointers without alteration.
EPA01_PUBLIC mjtNum mjc_ccd(const mjCCDConfig* config, mjCCDStatus* status,
                            mjCCDObj* obj1, mjCCDObj* obj2) {
  if (atomic_load_explicit(&armed, memory_order_acquire) != 1
      || atomic_load_explicit(&phase, memory_order_acquire) == 0) {
    hard_stop("mjc_ccd outside armed construction/control phase");
  }
  atomic_fetch_add_explicit(&ccd_attempts, 1, memory_order_relaxed);
  remember_first(&first_ccd_caller, (uintptr_t)__builtin_return_address(0));
  mjtNum result = epa01_engine_local_ccd(config, status, obj1, obj2);
  atomic_fetch_add_explicit(&ccd_returns, 1, memory_order_release);
  return result;
}

EPA01_PUBLIC void mj_step(const mjModel* model, mjData* data) {
  if (atomic_load_explicit(&armed, memory_order_acquire) != 1) {
    hard_stop("mj_step before arm");
  }
  int current_phase = atomic_load_explicit(&phase, memory_order_acquire);
  if (!model || !data) hard_stop("mj_step with null model/data");
  if (current_phase == 1) {
    reserve_step(&construction_attempts, &configured_construction_limit,
                 "construction mj_step budget exceeded");
    remember_first(&first_construction_step_caller,
                   (uintptr_t)__builtin_return_address(0));
    real_step()(model, data);
    atomic_fetch_add_explicit(&construction_returns, 1, memory_order_release);
  } else if (current_phase == 2) {
    if ((uintptr_t)model != atomic_load_explicit(&target_model, memory_order_acquire)
        || (uintptr_t)data != atomic_load_explicit(&target_data, memory_order_acquire)) {
      hard_stop("control mj_step target mismatch");
    }
    reserve_step(&control_attempts, &configured_control_limit,
                 "control mj_step budget exceeded");
    remember_first(&first_control_step_caller,
                   (uintptr_t)__builtin_return_address(0));
    real_step()(model, data);
    atomic_fetch_add_explicit(&control_returns, 1, memory_order_release);
  } else {
    hard_stop("mj_step outside construction/control phase");
  }
}
