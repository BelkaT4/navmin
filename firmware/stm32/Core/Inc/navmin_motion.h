#ifndef NAVMIN_MOTION_H
#define NAVMIN_MOTION_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAVMIN_CONTROL_RATE_HZ UINT32_C(20000)
#define NAVMIN_VELOCITY_SCALE UINT32_C(20000)
#define NAVMIN_STEP_PHASE_THRESHOLD INT32_C(400000000)

typedef enum {
    NAVMIN_MOTION_AXIS_X = 0,
    NAVMIN_MOTION_AXIS_Y = 1,
} navmin_motion_axis_t;

typedef enum {
    NAVMIN_STEP_DIRECTION_NEGATIVE = -1,
    NAVMIN_STEP_DIRECTION_POSITIVE = 1,
} navmin_step_direction_t;

typedef void (*navmin_emit_step_fn)(
    void *context,
    navmin_motion_axis_t axis,
    navmin_step_direction_t direction
);

typedef struct {
    void *context;
    navmin_emit_step_fn emit_step;
} navmin_motion_step_sink_t;

typedef struct {
    uint32_t max_speed_x_steps_s;
    uint32_t max_speed_y_steps_s;
    uint32_t acceleration_x_steps_s2;
    uint32_t acceleration_y_steps_s2;
} navmin_motion_limits_t;

typedef struct {
    /* Fixed-point velocity unit: 1 / NAVMIN_VELOCITY_SCALE steps/s. */
    int32_t current_velocity_q;
    int32_t target_velocity_q;

    /* Integrated displacement numerator. One STEP is emitted at
       +/-NAVMIN_STEP_PHASE_THRESHOLD. */
    int32_t step_phase_q;
    int64_t commanded_position_steps;
} navmin_motion_axis_state_t;

typedef struct {
    navmin_motion_axis_state_t x;
    navmin_motion_axis_state_t y;
    navmin_motion_limits_t limits;
    navmin_motion_step_sink_t step_sink;
    bool limits_valid;
} navmin_motion_t;

void navmin_motion_init(
    navmin_motion_t *motion,
    navmin_motion_step_sink_t step_sink
);

bool navmin_motion_set_limits(
    navmin_motion_t *motion,
    navmin_motion_limits_t limits
);

void navmin_motion_set_velocity_target(
    navmin_motion_t *motion,
    int32_t velocity_x_steps_s,
    int32_t velocity_y_steps_s
);

void navmin_motion_control_tick(navmin_motion_t *motion);

void navmin_motion_hard_stop(navmin_motion_t *motion);

#ifdef __cplusplus
}
#endif

#endif
