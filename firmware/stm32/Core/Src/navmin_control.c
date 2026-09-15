#include "navmin_control.h"

#include <limits.h>
#include <string.h>

#define NAVMIN_SET_CONFIG_PAYLOAD_LENGTH UINT8_C(20)
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
    case NAVMIN_COMMAND_SET_BAUDRATE:
        /* Recognized protocol commands whose owners are intentionally deferred
           to 4B3/4C. Never report fake success from this control owner. */
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

    if (control->velocity_watchdog_armed) {
        uint32_t elapsed = now_ms - control->last_velocity_setpoint_ms;

        if (elapsed >= control->config.velocity_watchdog_timeout_ms) {
            navmin_motion_set_velocity_target(&control->motion, 0, 0);
            control->velocity_watchdog_armed = false;
        }
    }

    navmin_motion_control_tick(&control->motion);
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
