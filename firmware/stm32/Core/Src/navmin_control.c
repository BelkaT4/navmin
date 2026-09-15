#include "navmin_control.h"

#include <limits.h>
#include <string.h>

#define NAVMIN_SET_CONFIG_PAYLOAD_LENGTH UINT8_C(20)
#define NAVMIN_MOVE_RELATIVE_PAYLOAD_LENGTH UINT8_C(8)
#define NAVMIN_SET_VELOCITY_PAYLOAD_LENGTH UINT8_C(8)

static uint32_t read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static int32_t read_i32_le(const uint8_t *bytes)
{
    uint32_t raw = read_u32_le(bytes);

    if (raw <= (uint32_t)INT32_MAX) {
        return (int32_t)raw;
    }
    return (int32_t)((int64_t)raw - INT64_C(4294967296));
}

static bool relative_delta_in_bounds(int32_t delta, int32_t limit)
{
    int64_t magnitude = delta;

    if (magnitude < 0) {
        magnitude = -magnitude;
    }
    return magnitude <= limit;
}

static bool add_i64_i32_checked(int64_t base, int32_t delta, int64_t *result)
{
    int64_t delta_i64 = delta;

    if ((delta_i64 > 0 && base > INT64_MAX - delta_i64) ||
        (delta_i64 < 0 && base < INT64_MIN - delta_i64)) {
        return false;
    }
    *result = base + delta_i64;
    return true;
}

static uint64_t position_distance(int64_t first, int64_t second)
{
    if (first >= second) {
        return (uint64_t)first - (uint64_t)second;
    }
    return (uint64_t)second - (uint64_t)first;
}

static uint64_t braking_steps_in_current_direction(
    const navmin_motion_axis_state_t *axis,
    uint32_t acceleration_steps_s2
)
{
    int32_t current = axis->current_velocity_q;
    uint64_t velocity_magnitude;
    uint64_t acceleration = acceleration_steps_s2;
    uint64_t positive_updates;
    uint64_t future_phase;
    int64_t directed_phase;
    int64_t total_phase;

    if (current == 0) {
        return 0U;
    }

    velocity_magnitude = current > 0 ?
        (uint64_t)current : (uint64_t)(-(int64_t)current);
    positive_updates = (velocity_magnitude - UINT64_C(1)) / acceleration;
    future_phase =
        positive_updates * velocity_magnitude -
        acceleration * positive_updates * (positive_updates + UINT64_C(1)) /
            UINT64_C(2);

    directed_phase = current > 0 ?
        (int64_t)axis->step_phase_q : -(int64_t)axis->step_phase_q;
    total_phase = directed_phase + (int64_t)future_phase;
    if (total_phase <= 0) {
        return 0U;
    }
    return (uint64_t)total_phase / (uint64_t)NAVMIN_STEP_PHASE_THRESHOLD;
}

static int32_t relative_planner_target_velocity(
    const navmin_motion_axis_state_t *axis,
    int64_t target_position_steps,
    uint32_t max_speed_steps_s,
    uint32_t acceleration_steps_s2
)
{
    int direction;
    uint64_t remaining_steps;
    uint64_t braking_steps;

    if (axis->commanded_position_steps == target_position_steps) {
        return 0;
    }

    direction = axis->commanded_position_steps < target_position_steps ? 1 : -1;
    remaining_steps = position_distance(
        axis->commanded_position_steps,
        target_position_steps);

    if ((direction > 0 && axis->current_velocity_q < 0) ||
        (direction < 0 && axis->current_velocity_q > 0)) {
        return direction > 0 ?
            (int32_t)max_speed_steps_s : -(int32_t)max_speed_steps_s;
    }

    braking_steps = braking_steps_in_current_direction(
        axis,
        acceleration_steps_s2);
    if (braking_steps >= remaining_steps) {
        return 0;
    }

    return direction > 0 ?
        (int32_t)max_speed_steps_s : -(int32_t)max_speed_steps_s;
}

static bool relative_axis_settled(
    const navmin_motion_axis_state_t *axis,
    int64_t target_position_steps
)
{
    return axis->commanded_position_steps == target_position_steps &&
           axis->current_velocity_q == 0 &&
           axis->target_velocity_q == 0 &&
           axis->step_phase_q == 0;
}

static void enter_critical(navmin_control_t *control)
{
    if (control->hardware.enter_critical != NULL) {
        control->hardware.enter_critical(control->hardware.context);
    }
}

static void exit_critical(navmin_control_t *control)
{
    if (control->hardware.exit_critical != NULL) {
        control->hardware.exit_critical(control->hardware.context);
    }
}

static void hard_stop_motion(navmin_control_t *control)
{
    enter_critical(control);
    navmin_motion_hard_stop(&control->motion);
    control->velocity_target_present = false;
    control->relative_target_present = false;
    control->velocity_watchdog_armed = false;
    exit_critical(control);
}

static bool config_is_valid(const navmin_control_config_t *config)
{
    return config->max_speed_x_steps_s > 0U &&
           config->max_speed_x_steps_s <= NAVMIN_MAX_SUPPORTED_STEP_RATE &&
           config->max_speed_y_steps_s > 0U &&
           config->max_speed_y_steps_s <= NAVMIN_MAX_SUPPORTED_STEP_RATE &&
           config->acceleration_x_steps_s2 > 0U &&
           config->acceleration_x_steps_s2 <= NAVMIN_MAX_SUPPORTED_ACCELERATION &&
           config->acceleration_y_steps_s2 > 0U &&
           config->acceleration_y_steps_s2 <= NAVMIN_MAX_SUPPORTED_ACCELERATION &&
           config->velocity_watchdog_timeout_ms > 0U &&
           config->velocity_watchdog_timeout_ms <=
               NAVMIN_MAX_SUPPORTED_WATCHDOG_TIMEOUT_MS;
}

static navmin_motion_limits_t motion_limits_from_config(
    const navmin_control_config_t *config
)
{
    navmin_motion_limits_t limits;

    limits.max_speed_x_steps_s = config->max_speed_x_steps_s;
    limits.max_speed_y_steps_s = config->max_speed_y_steps_s;
    limits.acceleration_x_steps_s2 = config->acceleration_x_steps_s2;
    limits.acceleration_y_steps_s2 = config->acceleration_y_steps_s2;
    return limits;
}

static navmin_result_code_t set_config(
    navmin_control_t *control,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    navmin_control_config_t next;
    navmin_motion_limits_t next_limits;

    if (payload_length != NAVMIN_SET_CONFIG_PAYLOAD_LENGTH || payload == NULL) {
        return NAVMIN_RESULT_PARSE_ERROR;
    }

    next.max_speed_x_steps_s = read_u32_le(&payload[0]);
    next.max_speed_y_steps_s = read_u32_le(&payload[4]);
    next.acceleration_x_steps_s2 = read_u32_le(&payload[8]);
    next.acceleration_y_steps_s2 = read_u32_le(&payload[12]);
    next.velocity_watchdog_timeout_ms = read_u32_le(&payload[16]);

    if (!config_is_valid(&next)) {
        return NAVMIN_RESULT_INVALID_ARGUMENT;
    }

    next_limits = motion_limits_from_config(&next);

    /* Future timer/ISR readers must see one coherent replacement: motion
       limits (including target clipping) and the matching control snapshot
       are published under the same hardware-independent critical boundary. */
    enter_critical(control);
    if (!navmin_motion_set_limits(&control->motion, next_limits)) {
        exit_critical(control);
        return NAVMIN_RESULT_INTERNAL_ERROR;
    }
    control->config = next;
    control->configured = true;
    exit_critical(control);
    return NAVMIN_RESULT_OK;
}

static navmin_result_code_t move_relative(
    navmin_control_t *control,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    int32_t delta_x_steps;
    int32_t delta_y_steps;
    int64_t target_x_steps;
    int64_t target_y_steps;

    if (payload_length != NAVMIN_MOVE_RELATIVE_PAYLOAD_LENGTH || payload == NULL) {
        return NAVMIN_RESULT_PARSE_ERROR;
    }

    delta_x_steps = read_i32_le(&payload[0]);
    delta_y_steps = read_i32_le(&payload[4]);

    if (!relative_delta_in_bounds(
            delta_x_steps,
            NAVMIN_MAX_RELATIVE_DELTA_X_STEPS) ||
        !relative_delta_in_bounds(
            delta_y_steps,
            NAVMIN_MAX_RELATIVE_DELTA_Y_STEPS)) {
        return NAVMIN_RESULT_INVALID_ARGUMENT;
    }
    if (!control->configured) {
        return NAVMIN_RESULT_NOT_CONFIGURED;
    }
    if (!control->motors_on) {
        return NAVMIN_RESULT_MOTORS_OFF;
    }

    enter_critical(control);
    if (!add_i64_i32_checked(
            control->motion.x.commanded_position_steps,
            delta_x_steps,
            &target_x_steps) ||
        !add_i64_i32_checked(
            control->motion.y.commanded_position_steps,
            delta_y_steps,
            &target_y_steps)) {
        exit_critical(control);
        return NAVMIN_RESULT_INTERNAL_ERROR;
    }

    control->relative_target_x_steps = target_x_steps;
    control->relative_target_y_steps = target_y_steps;
    control->relative_target_present = true;
    control->velocity_target_present = false;
    control->velocity_watchdog_armed = false;
    exit_critical(control);
    return NAVMIN_RESULT_OK;
}

static navmin_result_code_t set_velocity(
    navmin_control_t *control,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
)
{
    int32_t velocity_x_steps_s;
    int32_t velocity_y_steps_s;

    if (payload_length != NAVMIN_SET_VELOCITY_PAYLOAD_LENGTH || payload == NULL) {
        return NAVMIN_RESULT_PARSE_ERROR;
    }

    velocity_x_steps_s = read_i32_le(&payload[0]);
    velocity_y_steps_s = read_i32_le(&payload[4]);

    if (!control->configured) {
        return NAVMIN_RESULT_NOT_CONFIGURED;
    }
    if (!control->motors_on) {
        return NAVMIN_RESULT_MOTORS_OFF;
    }

    enter_critical(control);
    navmin_motion_set_velocity_target(
        &control->motion,
        velocity_x_steps_s,
        velocity_y_steps_s);
    control->relative_target_present = false;
    control->velocity_target_present = true;
    control->last_velocity_setpoint_ms = now_ms;
    control->velocity_watchdog_armed = true;
    exit_critical(control);
    return NAVMIN_RESULT_OK;
}

static navmin_result_code_t motor_on(
    navmin_control_t *control,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    (void)payload;
    if (payload_length != 0U) {
        return NAVMIN_RESULT_PARSE_ERROR;
    }
    if (!control->configured) {
        return NAVMIN_RESULT_NOT_CONFIGURED;
    }
    if (!control->motors_on) {
        if (control->hardware.set_drivers_enabled != NULL) {
            control->hardware.set_drivers_enabled(control->hardware.context, true);
        }
        control->motors_on = true;
    }
    return NAVMIN_RESULT_OK;
}

static navmin_result_code_t motor_off(
    navmin_control_t *control,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    (void)payload;
    if (payload_length != 0U) {
        return NAVMIN_RESULT_PARSE_ERROR;
    }

    hard_stop_motion(control);
    if (control->motors_on) {
        if (control->hardware.set_drivers_enabled != NULL) {
            control->hardware.set_drivers_enabled(control->hardware.context, false);
        }
        control->motors_on = false;
    }
    return NAVMIN_RESULT_OK;
}

void navmin_control_init(
    navmin_control_t *control,
    navmin_control_hardware_t hardware
)
{
    navmin_motion_step_sink_t step_sink;

    memset(control, 0, sizeof(*control));
    control->hardware = hardware;

    step_sink.context = hardware.context;
    step_sink.emit_step = hardware.emit_step;
    navmin_motion_init(&control->motion, step_sink);
    hard_stop_motion(control);

    if (control->hardware.set_drivers_enabled != NULL) {
        control->hardware.set_drivers_enabled(control->hardware.context, false);
    }
}

navmin_result_code_t navmin_control_execute_command(
    void *context,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
)
{
    navmin_control_t *control = context;

    if (control == NULL) {
        return NAVMIN_RESULT_INTERNAL_ERROR;
    }

    switch (command_code) {
    case NAVMIN_COMMAND_SET_CONFIG:
        return set_config(control, payload, payload_length);
    case NAVMIN_COMMAND_MOTOR_ON:
        return motor_on(control, payload, payload_length);
    case NAVMIN_COMMAND_MOTOR_OFF:
        return motor_off(control, payload, payload_length);
    case NAVMIN_COMMAND_SET_VELOCITY:
        return set_velocity(control, payload, payload_length, now_ms);
    case NAVMIN_COMMAND_MOVE_RELATIVE:
        return move_relative(control, payload, payload_length);
    case NAVMIN_COMMAND_SET_BAUDRATE:
        /* Recognized protocol command whose owner is intentionally deferred
           to 4C. Never report fake success from this control owner. */
        return NAVMIN_RESULT_INTERNAL_ERROR;
    default:
        return NAVMIN_RESULT_INTERNAL_ERROR;
    }
}

void navmin_control_tick(navmin_control_t *control, uint32_t now_ms)
{
    if (control == NULL) {
        return;
    }

    if (control->relative_target_present) {
        int32_t target_velocity_x = relative_planner_target_velocity(
            &control->motion.x,
            control->relative_target_x_steps,
            control->motion.limits.max_speed_x_steps_s,
            control->motion.limits.acceleration_x_steps_s2);
        int32_t target_velocity_y = relative_planner_target_velocity(
            &control->motion.y,
            control->relative_target_y_steps,
            control->motion.limits.max_speed_y_steps_s,
            control->motion.limits.acceleration_y_steps_s2);

        navmin_motion_set_velocity_target(
            &control->motion,
            target_velocity_x,
            target_velocity_y);
    } else if (control->velocity_watchdog_armed) {
        uint32_t elapsed = now_ms - control->last_velocity_setpoint_ms;

        if (elapsed >= control->config.velocity_watchdog_timeout_ms) {
            navmin_motion_set_velocity_target(&control->motion, 0, 0);
            control->velocity_watchdog_armed = false;
        }
    }

    navmin_motion_control_tick(&control->motion);

    if (control->relative_target_present &&
        relative_axis_settled(
            &control->motion.x,
            control->relative_target_x_steps) &&
        relative_axis_settled(
            &control->motion.y,
            control->relative_target_y_steps)) {
        control->relative_target_present = false;
    }
}

void navmin_control_emergency_stop(void *context)
{
    navmin_control_t *control = context;

    if (control != NULL) {
        hard_stop_motion(control);
    }
}

navmin_protocol_executor_t navmin_control_protocol_executor(navmin_control_t *control)
{
    navmin_protocol_executor_t executor;

    executor.context = control;
    executor.emergency_stop = navmin_control_emergency_stop;
    executor.execute_command = navmin_control_execute_command;
    return executor;
}
