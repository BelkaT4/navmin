#include "navmin_motion.h"

#include "navmin_control.h"

#include <limits.h>
#include <string.h>

_Static_assert(
    NAVMIN_CONTROL_RATE_HZ == NAVMIN_VELOCITY_SCALE,
    "velocity scale relies on one scale unit per control tick denominator");
_Static_assert(
    NAVMIN_MAX_SUPPORTED_STEP_RATE <=
        (uint32_t)(INT32_MAX / (int32_t)NAVMIN_VELOCITY_SCALE),
    "maximum velocity fixed-point representation must fit int32_t");
_Static_assert(
    NAVMIN_STEP_PHASE_THRESHOLD ==
        (int32_t)(NAVMIN_CONTROL_RATE_HZ * NAVMIN_VELOCITY_SCALE),
    "STEP phase threshold must represent exactly one step");
_Static_assert(
    NAVMIN_MAX_SUPPORTED_STEP_RATE * NAVMIN_VELOCITY_SCALE <
        (uint32_t)NAVMIN_STEP_PHASE_THRESHOLD,
    "scheduler assumes less than one whole step of velocity phase per tick");

static bool limits_are_valid(const navmin_motion_limits_t *limits)
{
    return limits->max_speed_x_steps_s > 0U &&
           limits->max_speed_x_steps_s <= NAVMIN_MAX_SUPPORTED_STEP_RATE &&
           limits->max_speed_y_steps_s > 0U &&
           limits->max_speed_y_steps_s <= NAVMIN_MAX_SUPPORTED_STEP_RATE &&
           limits->acceleration_x_steps_s2 > 0U &&
           limits->acceleration_x_steps_s2 <= NAVMIN_MAX_SUPPORTED_ACCELERATION &&
           limits->acceleration_y_steps_s2 > 0U &&
           limits->acceleration_y_steps_s2 <= NAVMIN_MAX_SUPPORTED_ACCELERATION;
}

static int32_t clamp_target_q(int32_t requested_steps_s, uint32_t max_speed_steps_s)
{
    int64_t requested = requested_steps_s;
    int64_t limit = max_speed_steps_s;

    if (requested > limit) {
        requested = limit;
    } else if (requested < -limit) {
        requested = -limit;
    }

    return (int32_t)(requested * (int64_t)NAVMIN_VELOCITY_SCALE);
}

static void clamp_existing_target(
    navmin_motion_axis_state_t *axis,
    uint32_t max_speed_steps_s
)
{
    int32_t max_q = (int32_t)(max_speed_steps_s * NAVMIN_VELOCITY_SCALE);

    if (axis->target_velocity_q > max_q) {
        axis->target_velocity_q = max_q;
    } else if (axis->target_velocity_q < -max_q) {
        axis->target_velocity_q = -max_q;
    }
}

static int32_t advance_velocity(
    int32_t current,
    int32_t target,
    uint32_t acceleration_steps_s2
)
{
    int32_t delta_q = (int32_t)acceleration_steps_s2;

    if (current == target) {
        return current;
    }

    /* A reversal consumes this tick only by braking toward exact zero. Any
       unused acceleration budget is deliberately discarded; opposite-sign
       acceleration starts on a later control tick. */
    if ((current > 0 && target < 0) || (current < 0 && target > 0)) {
        if (current > 0) {
            return current <= delta_q ? 0 : current - delta_q;
        }
        return current >= -delta_q ? 0 : current + delta_q;
    }

    if (current < target) {
        int32_t remaining = target - current;
        return remaining <= delta_q ? target : current + delta_q;
    }

    {
        int32_t remaining = current - target;
        return remaining <= delta_q ? target : current - delta_q;
    }
}

static void emit_step(
    navmin_motion_t *motion,
    navmin_motion_axis_t axis_id,
    navmin_motion_axis_state_t *axis,
    navmin_step_direction_t direction
)
{
    if (motion->step_sink.emit_step == NULL) {
        return;
    }

    motion->step_sink.emit_step(motion->step_sink.context, axis_id, direction);
    if (direction == NAVMIN_STEP_DIRECTION_POSITIVE) {
        ++axis->commanded_position_steps;
    } else {
        --axis->commanded_position_steps;
    }
}

static void update_axis(
    navmin_motion_t *motion,
    navmin_motion_axis_t axis_id,
    navmin_motion_axis_state_t *axis,
    uint32_t acceleration_steps_s2
)
{
    axis->current_velocity_q = advance_velocity(
        axis->current_velocity_q,
        axis->target_velocity_q,
        acceleration_steps_s2);

    if (axis->current_velocity_q == 0) {
        /* Fractional displacement from a stopped command is intentionally not
           carried into a later command or across a reversal. */
        axis->step_phase_q = 0;
        return;
    }

    axis->step_phase_q += axis->current_velocity_q;

    if (axis->step_phase_q >= NAVMIN_STEP_PHASE_THRESHOLD) {
        emit_step(motion, axis_id, axis, NAVMIN_STEP_DIRECTION_POSITIVE);
        axis->step_phase_q -= NAVMIN_STEP_PHASE_THRESHOLD;
    } else if (axis->step_phase_q <= -NAVMIN_STEP_PHASE_THRESHOLD) {
        emit_step(motion, axis_id, axis, NAVMIN_STEP_DIRECTION_NEGATIVE);
        axis->step_phase_q += NAVMIN_STEP_PHASE_THRESHOLD;
    }
}

void navmin_motion_init(
    navmin_motion_t *motion,
    navmin_motion_step_sink_t step_sink
)
{
    memset(motion, 0, sizeof(*motion));
    motion->step_sink = step_sink;
}

bool navmin_motion_set_limits(
    navmin_motion_t *motion,
    navmin_motion_limits_t limits
)
{
    if (!limits_are_valid(&limits)) {
        return false;
    }

    motion->limits = limits;
    motion->limits_valid = true;

    /* A lower runtime speed limit clips only the currently stored effective
       target. Current velocity remains continuous and reaches it through the
       limiter. Raising the limit never resurrects an older unclamped target. */
    clamp_existing_target(&motion->x, limits.max_speed_x_steps_s);
    clamp_existing_target(&motion->y, limits.max_speed_y_steps_s);
    return true;
}

void navmin_motion_set_velocity_target(
    navmin_motion_t *motion,
    int32_t velocity_x_steps_s,
    int32_t velocity_y_steps_s
)
{
    if (!motion->limits_valid) {
        motion->x.target_velocity_q = 0;
        motion->y.target_velocity_q = 0;
        return;
    }

    motion->x.target_velocity_q = clamp_target_q(
        velocity_x_steps_s,
        motion->limits.max_speed_x_steps_s);
    motion->y.target_velocity_q = clamp_target_q(
        velocity_y_steps_s,
        motion->limits.max_speed_y_steps_s);
}

void navmin_motion_control_tick(navmin_motion_t *motion)
{
    if (!motion->limits_valid) {
        return;
    }

    update_axis(
        motion,
        NAVMIN_MOTION_AXIS_X,
        &motion->x,
        motion->limits.acceleration_x_steps_s2);
    update_axis(
        motion,
        NAVMIN_MOTION_AXIS_Y,
        &motion->y,
        motion->limits.acceleration_y_steps_s2);
}

void navmin_motion_hard_stop(navmin_motion_t *motion)
{
    motion->x.current_velocity_q = 0;
    motion->x.target_velocity_q = 0;
    motion->x.step_phase_q = 0;
    motion->y.current_velocity_q = 0;
    motion->y.target_velocity_q = 0;
    motion->y.step_phase_q = 0;
}
