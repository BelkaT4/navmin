#include "navmin_control.h"
#include "navmin_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RESPONSE_CAPACITY 16U

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #expr); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ASSERT_EQ_U32(expected, actual) do { \
    uint32_t expected_value_ = (uint32_t)(expected); \
    uint32_t actual_value_ = (uint32_t)(actual); \
    if (expected_value_ != actual_value_) { \
        fprintf(stderr, "%s:%d: expected %lu, got %lu\n", \
                __FILE__, __LINE__, \
                (unsigned long)expected_value_, (unsigned long)actual_value_); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ASSERT_EQ_I32(expected, actual) do { \
    int32_t expected_value_ = (int32_t)(expected); \
    int32_t actual_value_ = (int32_t)(actual); \
    if (expected_value_ != actual_value_) { \
        fprintf(stderr, "%s:%d: expected %ld, got %ld\n", \
                __FILE__, __LINE__, \
                (long)expected_value_, (long)actual_value_); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ASSERT_EQ_I64(expected, actual) do { \
    int64_t expected_value_ = (int64_t)(expected); \
    int64_t actual_value_ = (int64_t)(actual); \
    if (expected_value_ != actual_value_) { \
        fprintf(stderr, "%s:%d: expected %lld, got %lld\n", \
                __FILE__, __LINE__, \
                (long long)expected_value_, (long long)actual_value_); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef struct {
    bool enabled;
    unsigned call_count;
    unsigned enable_count;
    unsigned disable_count;
    unsigned step_count;
    unsigned positive_step_count;
    unsigned negative_step_count;
    unsigned x_positive_step_count;
    unsigned x_negative_step_count;
    unsigned y_positive_step_count;
    unsigned y_negative_step_count;
    unsigned critical_enter_count;
    unsigned critical_exit_count;
    unsigned critical_depth;
    unsigned max_critical_depth;
    navmin_control_t *observed_control;
    bool observe_hard_stop;
    bool hard_stop_active_at_enter;
    bool hard_stop_cleared_at_exit;
    bool drivers_enabled_at_hard_stop_exit;
    int64_t expected_x_position;
    int64_t expected_y_position;
} driver_sink_t;

typedef struct {
    uint8_t bytes[RESPONSE_CAPACITY][NAVMIN_MIN_RESPONSE_LENGTH];
    unsigned count;
} response_log_t;

typedef struct {
    navmin_control_t *control;
    unsigned execute_count;
    unsigned emergency_count;
} executor_probe_t;

static void write_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t)(value & UINT16_C(0x00FF));
    bytes[1] = (uint8_t)(value >> 8);
}

static void write_u32_le(uint8_t *bytes, uint32_t value)
{
    bytes[0] = (uint8_t)(value & UINT32_C(0x000000FF));
    bytes[1] = (uint8_t)((value >> 8) & UINT32_C(0x000000FF));
    bytes[2] = (uint8_t)((value >> 16) & UINT32_C(0x000000FF));
    bytes[3] = (uint8_t)(value >> 24);
}

static void set_drivers_enabled(void *context, bool enabled)
{
    driver_sink_t *sink = context;

    sink->enabled = enabled;
    ++sink->call_count;
    if (enabled) {
        ++sink->enable_count;
    } else {
        ++sink->disable_count;
    }
}

static void emit_step(
    void *context,
    navmin_motion_axis_t axis,
    navmin_step_direction_t direction
)
{
    driver_sink_t *sink = context;

    ++sink->step_count;
    if (direction == NAVMIN_STEP_DIRECTION_POSITIVE) {
        ++sink->positive_step_count;
        if (axis == NAVMIN_MOTION_AXIS_X) {
            ++sink->x_positive_step_count;
        } else {
            ++sink->y_positive_step_count;
        }
    } else {
        ++sink->negative_step_count;
        if (axis == NAVMIN_MOTION_AXIS_X) {
            ++sink->x_negative_step_count;
        } else {
            ++sink->y_negative_step_count;
        }
    }
}

static void enter_critical(void *context)
{
    driver_sink_t *sink = context;

    ++sink->critical_enter_count;
    ++sink->critical_depth;
    if (sink->critical_depth > sink->max_critical_depth) {
        sink->max_critical_depth = sink->critical_depth;
    }

    if (sink->observe_hard_stop) {
        navmin_control_t *control = sink->observed_control;

        ASSERT_TRUE(control != NULL);
        sink->hard_stop_active_at_enter =
            control->motion.x.current_velocity_q != 0 &&
            control->motion.x.target_velocity_q != 0 &&
            control->motion.x.step_phase_q != 0 &&
            control->motion.y.current_velocity_q != 0 &&
            control->motion.y.target_velocity_q != 0 &&
            control->motion.y.step_phase_q != 0 &&
            control->velocity_target_present &&
            control->relative_target_present;
    }
}

static void exit_critical(void *context)
{
    driver_sink_t *sink = context;

    ASSERT_EQ_U32(1U, sink->critical_depth);
    if (sink->observe_hard_stop) {
        const navmin_control_t *control = sink->observed_control;

        ASSERT_TRUE(control != NULL);
        sink->hard_stop_cleared_at_exit =
            control->motion.x.current_velocity_q == 0 &&
            control->motion.x.target_velocity_q == 0 &&
            control->motion.x.step_phase_q == 0 &&
            control->motion.y.current_velocity_q == 0 &&
            control->motion.y.target_velocity_q == 0 &&
            control->motion.y.step_phase_q == 0 &&
            !control->velocity_target_present &&
            !control->relative_target_present &&
            control->motion.x.commanded_position_steps ==
                sink->expected_x_position &&
            control->motion.y.commanded_position_steps ==
                sink->expected_y_position;
        sink->drivers_enabled_at_hard_stop_exit = sink->enabled;
    }
    ++sink->critical_exit_count;
    --sink->critical_depth;
}

static navmin_control_t make_control(driver_sink_t *sink)
{
    navmin_control_t control;
    navmin_control_hardware_t hardware;

    memset(sink, 0, sizeof(*sink));
    memset(&hardware, 0, sizeof(hardware));
    hardware.context = sink;
    hardware.set_drivers_enabled = set_drivers_enabled;
    hardware.emit_step = emit_step;
    hardware.enter_critical = enter_critical;
    hardware.exit_critical = exit_critical;
    navmin_control_init(&control, hardware);

    /* Init itself uses the common critical hard-stop. Operation tests measure
       only the boundary they invoke after make_control() returns. */
    sink->critical_enter_count = 0U;
    sink->critical_exit_count = 0U;
    sink->critical_depth = 0U;
    sink->max_critical_depth = 0U;
    return control;
}

static void build_config_payload(
    uint8_t payload[20],
    uint32_t max_speed_x,
    uint32_t max_speed_y,
    uint32_t acceleration_x,
    uint32_t acceleration_y,
    uint32_t watchdog_ms
)
{
    write_u32_le(&payload[0], max_speed_x);
    write_u32_le(&payload[4], max_speed_y);
    write_u32_le(&payload[8], acceleration_x);
    write_u32_le(&payload[12], acceleration_y);
    write_u32_le(&payload[16], watchdog_ms);
}

static void build_velocity_payload(
    uint8_t payload[8],
    int32_t velocity_x_steps_s,
    int32_t velocity_y_steps_s
)
{
    write_u32_le(&payload[0], (uint32_t)velocity_x_steps_s);
    write_u32_le(&payload[4], (uint32_t)velocity_y_steps_s);
}

static void build_relative_payload(
    uint8_t payload[8],
    int32_t delta_x_steps,
    int32_t delta_y_steps
)
{
    write_u32_le(&payload[0], (uint32_t)delta_x_steps);
    write_u32_le(&payload[4], (uint32_t)delta_y_steps);
}

static navmin_result_code_t set_velocity_direct(
    navmin_control_t *control,
    int32_t velocity_x_steps_s,
    int32_t velocity_y_steps_s,
    uint32_t now_ms
)
{
    uint8_t payload[8];

    build_velocity_payload(payload, velocity_x_steps_s, velocity_y_steps_s);
    return navmin_control_execute_command(
        control,
        NAVMIN_COMMAND_SET_VELOCITY,
        payload,
        (uint8_t)sizeof(payload),
        now_ms);
}

static navmin_result_code_t move_relative_direct(
    navmin_control_t *control,
    int32_t delta_x_steps,
    int32_t delta_y_steps
)
{
    uint8_t payload[8];

    build_relative_payload(payload, delta_x_steps, delta_y_steps);
    return navmin_control_execute_command(
        control,
        NAVMIN_COMMAND_MOVE_RELATIVE,
        payload,
        (uint8_t)sizeof(payload),
        0U);
}

static navmin_result_code_t apply_config(
    navmin_control_t *control,
    uint32_t max_speed_x,
    uint32_t max_speed_y,
    uint32_t acceleration_x,
    uint32_t acceleration_y,
    uint32_t watchdog_ms
)
{
    uint8_t payload[20];

    build_config_payload(
        payload,
        max_speed_x,
        max_speed_y,
        acceleration_x,
        acceleration_y,
        watchdog_ms);
    return navmin_control_execute_command(
        control,
        NAVMIN_COMMAND_SET_CONFIG,
        payload,
        (uint8_t)sizeof(payload),
        0U);
}

static void configure_and_motor_on(
    navmin_control_t *control,
    uint32_t max_speed,
    uint32_t acceleration,
    uint32_t watchdog_ms
)
{
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(
            control,
            max_speed,
            max_speed,
            acceleration,
            acceleration,
            watchdog_ms));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
}

static unsigned run_until_relative_settled(
    navmin_control_t *control,
    unsigned max_ticks
)
{
    unsigned tick;

    for (tick = 0U; tick < max_ticks; ++tick) {
        navmin_control_tick(control, 0U);
        if (!control->relative_target_present) {
            return tick + 1U;
        }
    }

    fprintf(stderr, "%s:%d: relative planner did not settle in %u ticks\n",
            __FILE__, __LINE__, max_ticks);
    exit(EXIT_FAILURE);
}

static void assert_relative_axis_settled(
    const navmin_motion_axis_state_t *axis,
    int64_t expected_position
)
{
    ASSERT_EQ_I64(expected_position, axis->commanded_position_steps);
    ASSERT_EQ_I32(0, axis->current_velocity_q);
    ASSERT_EQ_I32(0, axis->target_velocity_q);
    ASSERT_EQ_I32(0, axis->step_phase_q);
}

static void seed_motion_state(navmin_control_t *control)
{
    control->velocity_target_present = true;
    control->relative_target_present = true;

    control->motion.x.current_velocity_q =
        (int32_t)(700U * NAVMIN_VELOCITY_SCALE);
    control->motion.x.target_velocity_q =
        (int32_t)(800U * NAVMIN_VELOCITY_SCALE);
    control->motion.x.step_phase_q = 12345;
    control->motion.x.commanded_position_steps = 321;

    control->motion.y.current_velocity_q =
        -(int32_t)(500U * NAVMIN_VELOCITY_SCALE);
    control->motion.y.target_velocity_q =
        -(int32_t)(600U * NAVMIN_VELOCITY_SCALE);
    control->motion.y.step_phase_q = -23456;
    control->motion.y.commanded_position_steps = -654;
}

static void observe_hard_stop_boundary(
    driver_sink_t *sink,
    navmin_control_t *control
)
{
    ASSERT_EQ_U32(0U, sink->critical_depth);
    sink->critical_enter_count = 0U;
    sink->critical_exit_count = 0U;
    sink->max_critical_depth = 0U;
    sink->observed_control = control;
    sink->observe_hard_stop = true;
    sink->hard_stop_active_at_enter = false;
    sink->hard_stop_cleared_at_exit = false;
    sink->drivers_enabled_at_hard_stop_exit = false;
    sink->expected_x_position = control->motion.x.commanded_position_steps;
    sink->expected_y_position = control->motion.y.commanded_position_steps;
}

static void assert_hard_stop_boundary_observed(const driver_sink_t *sink)
{
    ASSERT_TRUE(sink->hard_stop_active_at_enter);
    ASSERT_TRUE(sink->hard_stop_cleared_at_exit);
    ASSERT_TRUE(sink->drivers_enabled_at_hard_stop_exit);
    ASSERT_EQ_U32(1U, sink->critical_enter_count);
    ASSERT_EQ_U32(1U, sink->critical_exit_count);
    ASSERT_EQ_U32(0U, sink->critical_depth);
    ASSERT_EQ_U32(1U, sink->max_critical_depth);
}

static void assert_motion_cleared(const navmin_control_t *control)
{
    ASSERT_TRUE(!control->velocity_target_present);
    ASSERT_TRUE(!control->relative_target_present);
}

static void assert_numeric_motion_hard_stopped(
    const navmin_control_t *control,
    int64_t expected_x_position,
    int64_t expected_y_position
)
{
    ASSERT_EQ_U32(0U, control->motion.x.current_velocity_q);
    ASSERT_EQ_U32(0U, control->motion.x.target_velocity_q);
    ASSERT_EQ_U32(0U, control->motion.x.step_phase_q);
    ASSERT_EQ_U32(0U, control->motion.y.current_velocity_q);
    ASSERT_EQ_U32(0U, control->motion.y.target_velocity_q);
    ASSERT_EQ_U32(0U, control->motion.y.step_phase_q);
    ASSERT_TRUE(control->motion.x.commanded_position_steps == expected_x_position);
    ASSERT_TRUE(control->motion.y.commanded_position_steps == expected_y_position);
}

static size_t build_request(
    uint8_t *output,
    uint16_t request_id,
    uint8_t command,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    uint8_t length = (uint8_t)(NAVMIN_MIN_REQUEST_LENGTH + payload_length);
    uint16_t crc;

    output[0] = NAVMIN_START_BYTE_0;
    output[1] = NAVMIN_START_BYTE_1;
    output[2] = length;
    write_u16_le(&output[3], request_id);
    output[5] = command;
    if (payload_length > 0U) {
        memcpy(&output[6], payload, payload_length);
    }
    crc = navmin_crc16_modbus(&output[2], (size_t)length - 4U);
    write_u16_le(&output[length - 2U], crc);
    return length;
}

static void capture_response(
    void *context,
    const uint8_t *response,
    uint8_t response_length
)
{
    response_log_t *log = context;

    ASSERT_TRUE(log->count < RESPONSE_CAPACITY);
    ASSERT_EQ_U32(NAVMIN_MIN_RESPONSE_LENGTH, response_length);
    memcpy(log->bytes[log->count], response, response_length);
    ++log->count;
}

static void feed_frame(
    navmin_protocol_t *protocol,
    response_log_t *log,
    const uint8_t *frame,
    size_t length,
    uint32_t *now_ms
)
{
    size_t i;

    for (i = 0U; i < length; ++i) {
        navmin_protocol_feed_byte(
            protocol,
            frame[i],
            *now_ms,
            capture_response,
            log);
        ++*now_ms;
    }
}

static void feed_frame_ending_at(
    navmin_protocol_t *protocol,
    response_log_t *log,
    const uint8_t *frame,
    size_t length,
    uint32_t acceptance_now_ms
)
{
    uint32_t now_ms;

    ASSERT_TRUE(length > 0U);
    now_ms = acceptance_now_ms - (uint32_t)(length - 1U);
    feed_frame(protocol, log, frame, length, &now_ms);
}

static void assert_response_result(
    const response_log_t *log,
    unsigned index,
    navmin_result_code_t result
)
{
    ASSERT_TRUE(index < log->count);
    ASSERT_EQ_U32(result, log->bytes[index][6]);
}

static navmin_result_code_t probe_execute(
    void *context,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
)
{
    executor_probe_t *probe = context;

    ++probe->execute_count;
    return navmin_control_execute_command(
        probe->control,
        command_code,
        payload,
        payload_length,
        now_ms);
}

static void probe_emergency(void *context)
{
    executor_probe_t *probe = context;

    ++probe->emergency_count;
    navmin_control_emergency_stop(probe->control);
}

static void test_boot_is_unconfigured_and_motors_off(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_TRUE(!control.configured);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_TRUE(!sink.enabled);
    ASSERT_EQ_U32(1U, sink.disable_count);
    assert_motion_cleared(&control);
    ASSERT_TRUE(!control.motion.limits_valid);
    ASSERT_EQ_U32(0U, control.motion.x.current_velocity_q);
    ASSERT_EQ_U32(0U, control.motion.x.target_velocity_q);
    ASSERT_EQ_U32(0U, control.motion.x.step_phase_q);
    ASSERT_TRUE(control.motion.x.commanded_position_steps == 0);
    ASSERT_EQ_U32(0U, control.motion.y.current_velocity_q);
    ASSERT_EQ_U32(0U, control.motion.y.target_velocity_q);
    ASSERT_EQ_U32(0U, control.motion.y.step_phase_q);
    ASSERT_TRUE(control.motion.y.commanded_position_steps == 0);
}

static void test_set_config_decodes_exact_little_endian_values(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    navmin_result_code_t result;

    result = apply_config(&control, 0x1234U, 0x2345U, 0x5678U, 0x9ABCU, 0x0FEDU);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, result);
    ASSERT_TRUE(control.configured);
    ASSERT_EQ_U32(0x1234U, control.config.max_speed_x_steps_s);
    ASSERT_EQ_U32(0x2345U, control.config.max_speed_y_steps_s);
    ASSERT_EQ_U32(0x5678U, control.config.acceleration_x_steps_s2);
    ASSERT_EQ_U32(0x9ABCU, control.config.acceleration_y_steps_s2);
    ASSERT_EQ_U32(0x0FEDU, control.config.velocity_watchdog_timeout_ms);
    ASSERT_TRUE(control.motion.limits_valid);
    ASSERT_EQ_U32(0x1234U, control.motion.limits.max_speed_x_steps_s);
    ASSERT_EQ_U32(0x2345U, control.motion.limits.max_speed_y_steps_s);
    ASSERT_EQ_U32(0x5678U, control.motion.limits.acceleration_x_steps_s2);
    ASSERT_EQ_U32(0x9ABCU, control.motion.limits.acceleration_y_steps_s2);
}

static void test_set_config_rejects_each_zero_field(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    uint32_t fields[5] = {1000U, 1000U, 5000U, 5000U, 200U};
    unsigned i;

    for (i = 0U; i < 5U; ++i) {
        uint32_t saved = fields[i];
        navmin_result_code_t result;

        fields[i] = 0U;
        result = apply_config(
            &control, fields[0], fields[1], fields[2], fields[3], fields[4]);
        ASSERT_EQ_U32(NAVMIN_RESULT_INVALID_ARGUMENT, result);
        ASSERT_TRUE(!control.configured);
        fields[i] = saved;
    }
}

static void test_set_config_accepts_all_firmware_bounds(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(
            &control,
            NAVMIN_MAX_SUPPORTED_STEP_RATE,
            NAVMIN_MAX_SUPPORTED_STEP_RATE,
            NAVMIN_MAX_SUPPORTED_ACCELERATION,
            NAVMIN_MAX_SUPPORTED_ACCELERATION,
            NAVMIN_MAX_SUPPORTED_WATCHDOG_TIMEOUT_MS));
}

static void test_set_config_rejects_each_bound_plus_one(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_INVALID_ARGUMENT,
        apply_config(
            &control,
            NAVMIN_MAX_SUPPORTED_STEP_RATE + 1U,
            1000U,
            5000U,
            5000U,
            200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INVALID_ARGUMENT,
        apply_config(
            &control,
            1000U,
            1000U,
            NAVMIN_MAX_SUPPORTED_ACCELERATION + 1U,
            5000U,
            200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INVALID_ARGUMENT,
        apply_config(
            &control,
            1000U,
            1000U,
            5000U,
            5000U,
            NAVMIN_MAX_SUPPORTED_WATCHDOG_TIMEOUT_MS + 1U));
    ASSERT_TRUE(!control.configured);
}

static void test_invalid_config_keeps_previous_snapshot_and_valid_replaces_all(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    navmin_control_config_t original;
    navmin_motion_limits_t original_limits;
    navmin_motion_axis_state_t original_x;
    navmin_motion_axis_state_t original_y;
    unsigned critical_enters_before_invalid;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 2000U, 3000U, 4000U, 500U));
    navmin_motion_set_velocity_target(&control.motion, 900, -800);
    control.motion.x.current_velocity_q =
        (int32_t)(700U * NAVMIN_VELOCITY_SCALE);
    control.motion.y.current_velocity_q =
        -(int32_t)(600U * NAVMIN_VELOCITY_SCALE);
    control.motion.x.step_phase_q = 123;
    control.motion.y.step_phase_q = -456;

    original = control.config;
    original_limits = control.motion.limits;
    original_x = control.motion.x;
    original_y = control.motion.y;
    critical_enters_before_invalid = sink.critical_enter_count;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_INVALID_ARGUMENT,
        apply_config(&control, 0U, 2222U, 3333U, 4444U, 555U));
    ASSERT_TRUE(memcmp(&control.config, &original, sizeof(original)) == 0);
    ASSERT_TRUE(
        memcmp(&control.motion.limits, &original_limits, sizeof(original_limits)) == 0);
    ASSERT_TRUE(memcmp(&control.motion.x, &original_x, sizeof(original_x)) == 0);
    ASSERT_TRUE(memcmp(&control.motion.y, &original_y, sizeof(original_y)) == 0);
    ASSERT_TRUE(control.configured);
    ASSERT_EQ_U32(critical_enters_before_invalid, sink.critical_enter_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1111U, 2222U, 3333U, 4444U, 555U));
    ASSERT_EQ_U32(1111U, control.config.max_speed_x_steps_s);
    ASSERT_EQ_U32(2222U, control.config.max_speed_y_steps_s);
    ASSERT_EQ_U32(3333U, control.config.acceleration_x_steps_s2);
    ASSERT_EQ_U32(4444U, control.config.acceleration_y_steps_s2);
    ASSERT_EQ_U32(555U, control.config.velocity_watchdog_timeout_ms);
    ASSERT_EQ_U32(1111U, control.motion.limits.max_speed_x_steps_s);
    ASSERT_EQ_U32(2222U, control.motion.limits.max_speed_y_steps_s);
    ASSERT_EQ_U32(3333U, control.motion.limits.acceleration_x_steps_s2);
    ASSERT_EQ_U32(4444U, control.motion.limits.acceleration_y_steps_s2);
}

static void test_set_config_uses_one_critical_replacement_boundary(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(0U, sink.critical_enter_count);
    ASSERT_EQ_U32(0U, sink.critical_exit_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1234U, 2345U, 3456U, 4567U, 500U));

    ASSERT_EQ_U32(1U, sink.critical_enter_count);
    ASSERT_EQ_U32(1U, sink.critical_exit_count);
    ASSERT_EQ_U32(0U, sink.critical_depth);
    ASSERT_EQ_U32(1U, sink.max_critical_depth);
    ASSERT_EQ_U32(
        control.config.max_speed_x_steps_s,
        control.motion.limits.max_speed_x_steps_s);
    ASSERT_EQ_U32(
        control.config.max_speed_y_steps_s,
        control.motion.limits.max_speed_y_steps_s);
    ASSERT_EQ_U32(
        control.config.acceleration_x_steps_s2,
        control.motion.limits.acceleration_x_steps_s2);
    ASSERT_EQ_U32(
        control.config.acceleration_y_steps_s2,
        control.motion.limits.acceleration_y_steps_s2);
}

static void test_runtime_max_speed_decrease_and_increase_wire_motion_limits(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    int32_t current_before;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 8000U, 8000U, 50000U, 50000U, 200U));

    navmin_motion_set_velocity_target(&control.motion, 7000, 0);
    control.motion.x.current_velocity_q =
        (int32_t)(7000U * NAVMIN_VELOCITY_SCALE);
    current_before = control.motion.x.current_velocity_q;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 5000U, 8000U, 50000U, 50000U, 200U));

    ASSERT_EQ_U32((uint32_t)current_before, control.motion.x.current_velocity_q);
    ASSERT_EQ_U32(
        5000U * NAVMIN_VELOCITY_SCALE,
        control.motion.x.target_velocity_q);
    ASSERT_EQ_U32(5000U, control.config.max_speed_x_steps_s);
    ASSERT_EQ_U32(5000U, control.motion.limits.max_speed_x_steps_s);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 9000U, 8000U, 50000U, 50000U, 200U));

    ASSERT_EQ_U32((uint32_t)current_before, control.motion.x.current_velocity_q);
    ASSERT_EQ_U32(
        5000U * NAVMIN_VELOCITY_SCALE,
        control.motion.x.target_velocity_q);
    ASSERT_EQ_U32(9000U, control.motion.limits.max_speed_x_steps_s);
}

static void test_runtime_acceleration_change_applies_on_next_motion_tick(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    int32_t current_before;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 8000U, 8000U, 50000U, 50000U, 200U));

    navmin_motion_set_velocity_target(&control.motion, 1000, 0);
    control.motion.x.current_velocity_q =
        (int32_t)(100U * NAVMIN_VELOCITY_SCALE);
    current_before = control.motion.x.current_velocity_q;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 8000U, 8000U, 100000U, 50000U, 200U));

    ASSERT_EQ_U32((uint32_t)current_before, control.motion.x.current_velocity_q);
    ASSERT_EQ_U32(100000U, control.motion.limits.acceleration_x_steps_s2);

    navmin_motion_control_tick(&control.motion);
    ASSERT_EQ_U32(
        (uint32_t)(current_before + 100000),
        control.motion.x.current_velocity_q);
}

static void test_control_owned_motion_routes_step_to_hardware_boundary(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(
            &control,
            NAVMIN_MAX_SUPPORTED_STEP_RATE,
            NAVMIN_MAX_SUPPORTED_STEP_RATE,
            NAVMIN_MAX_SUPPORTED_ACCELERATION,
            NAVMIN_MAX_SUPPORTED_ACCELERATION,
            200U));

    navmin_motion_set_velocity_target(
        &control.motion,
        (int32_t)NAVMIN_MAX_SUPPORTED_STEP_RATE,
        0);
    control.motion.x.current_velocity_q =
        (int32_t)(NAVMIN_MAX_SUPPORTED_STEP_RATE * NAVMIN_VELOCITY_SCALE);

    navmin_motion_control_tick(&control.motion);
    navmin_motion_control_tick(&control.motion);

    ASSERT_EQ_U32(1U, sink.step_count);
    ASSERT_EQ_U32(1U, sink.positive_step_count);
    ASSERT_EQ_U32(0U, sink.negative_step_count);
    ASSERT_TRUE(control.motion.x.commanded_position_steps == 1);
}

static void test_set_config_is_allowed_while_motors_on(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned hardware_calls_before;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    hardware_calls_before = sink.call_count;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 2000U, 3000U, 6000U, 7000U, 300U));

    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
    ASSERT_EQ_U32(hardware_calls_before, sink.call_count);
    ASSERT_EQ_U32(2000U, control.config.max_speed_x_steps_s);
    ASSERT_EQ_U32(3000U, control.config.max_speed_y_steps_s);
    ASSERT_EQ_U32(6000U, control.config.acceleration_x_steps_s2);
    ASSERT_EQ_U32(7000U, control.config.acceleration_y_steps_s2);
    ASSERT_EQ_U32(300U, control.config.velocity_watchdog_timeout_ms);
}

static void test_motor_on_requires_config_then_is_idempotent(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_NOT_CONFIGURED,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    ASSERT_TRUE(!control.motors_on);
    ASSERT_EQ_U32(0U, sink.enable_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
    ASSERT_EQ_U32(1U, sink.enable_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    ASSERT_EQ_U32(1U, sink.enable_count);
}

static void test_motor_off_before_config_and_while_off_is_idempotent(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned calls_after_init = sink.call_count;

    seed_motion_state(&control);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U, 0U));
    assert_motion_cleared(&control);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_EQ_U32(calls_after_init, sink.call_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U, 0U));
    ASSERT_EQ_U32(calls_after_init, sink.call_count);
}

static void test_motor_off_while_on_hard_stops_and_disables(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    seed_motion_state(&control);
    observe_hard_stop_boundary(&sink, &control);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U, 0U));

    assert_hard_stop_boundary_observed(&sink);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_TRUE(!sink.enabled);
    ASSERT_EQ_U32(1U, sink.enable_count);
    ASSERT_EQ_U32(2U, sink.disable_count);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);
}

static void test_motor_off_then_motor_on_does_not_restore_numeric_motion(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));

    seed_motion_state(&control);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U, 0U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));

    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);
}

static void test_emergency_while_on_hard_stops_without_disabling_drivers(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned calls_before;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    seed_motion_state(&control);
    observe_hard_stop_boundary(&sink, &control);
    calls_before = sink.call_count;

    navmin_control_emergency_stop(&control);

    assert_hard_stop_boundary_observed(&sink);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
    ASSERT_EQ_U32(calls_before, sink.call_count);
}

static void test_emergency_before_config_while_off_is_repeatable_without_latch(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned calls_after_init = sink.call_count;

    seed_motion_state(&control);
    navmin_control_emergency_stop(&control);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);
    ASSERT_TRUE(!control.configured);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_EQ_U32(calls_after_init, sink.call_count);

    seed_motion_state(&control);
    navmin_control_emergency_stop(&control);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);
    ASSERT_EQ_U32(calls_after_init, sink.call_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    ASSERT_TRUE(control.motors_on);
}

static void test_set_velocity_validation_clamp_and_extremes(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    uint8_t payload[8];
    int32_t target_x_before;
    int32_t target_y_before;
    uint32_t timestamp_before;

    build_velocity_payload(payload, 100, -100);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_NOT_CONFIGURED,
        navmin_control_execute_command(
            &control,
            NAVMIN_COMMAND_SET_VELOCITY,
            payload,
            (uint8_t)sizeof(payload),
            10U));
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_TRUE(!control.velocity_target_present);
    ASSERT_EQ_U32(0U, control.last_velocity_setpoint_ms);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 50000U, 50000U, 500U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_MOTORS_OFF,
        navmin_control_execute_command(
            &control,
            NAVMIN_COMMAND_SET_VELOCITY,
            payload,
            (uint8_t)sizeof(payload),
            20U));
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_TRUE(!control.velocity_target_present);
    ASSERT_EQ_U32(0U, control.last_velocity_setpoint_ms);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    control.relative_target_present = true;
    sink.critical_enter_count = 0U;
    sink.critical_exit_count = 0U;
    sink.max_critical_depth = 0U;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        set_velocity_direct(&control, INT32_MAX, INT32_MIN, 123U));
    ASSERT_EQ_I32(
        (int32_t)(1000U * NAVMIN_VELOCITY_SCALE),
        control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(
        -(int32_t)(1000U * NAVMIN_VELOCITY_SCALE),
        control.motion.y.target_velocity_q);
    ASSERT_EQ_I32(0, control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(0, control.motion.y.current_velocity_q);
    ASSERT_TRUE(control.velocity_target_present);
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_U32(123U, control.last_velocity_setpoint_ms);
    ASSERT_EQ_U32(1U, sink.critical_enter_count);
    ASSERT_EQ_U32(1U, sink.critical_exit_count);
    ASSERT_EQ_U32(1U, sink.max_critical_depth);

    target_x_before = control.motion.x.target_velocity_q;
    target_y_before = control.motion.y.target_velocity_q;
    timestamp_before = control.last_velocity_setpoint_ms;
    ASSERT_EQ_U32(
        NAVMIN_RESULT_PARSE_ERROR,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_SET_VELOCITY, payload, 7U, 999U));
    ASSERT_EQ_I32(target_x_before, control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(target_y_before, control.motion.y.target_velocity_q);
    ASSERT_EQ_U32(timestamp_before, control.last_velocity_setpoint_ms);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_TRUE(control.velocity_target_present);
    ASSERT_TRUE(!control.relative_target_present);
}

static void test_set_velocity_zero_is_ordinary_setpoint(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    int32_t current_before;

    configure_and_motor_on(&control, 1000U, 50000U, 500U);
    control.motion.x.current_velocity_q =
        (int32_t)(500U * NAVMIN_VELOCITY_SCALE);
    current_before = control.motion.x.current_velocity_q;
    control.relative_target_present = true;

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, 0, 0, 500U));

    ASSERT_EQ_I32(0, control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(0, control.motion.y.target_velocity_q);
    ASSERT_EQ_I32(current_before, control.motion.x.current_velocity_q);
    ASSERT_TRUE(control.velocity_target_present);
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_U32(500U, control.last_velocity_setpoint_ms);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
}

static void test_control_tick_watchdog_boundary_and_rearm(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    int32_t current_before_expiry;

    configure_and_motor_on(&control, 1000U, 100000U, 500U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, 1000, 0, 100U));
    control.motion.x.current_velocity_q =
        (int32_t)(100U * NAVMIN_VELOCITY_SCALE);

    navmin_control_tick(&control, 599U);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_I32(
        (int32_t)(100U * NAVMIN_VELOCITY_SCALE) + 100000,
        control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(
        (int32_t)(1000U * NAVMIN_VELOCITY_SCALE),
        control.motion.x.target_velocity_q);

    current_before_expiry = control.motion.x.current_velocity_q;
    navmin_control_tick(&control, 600U);

    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_EQ_I32(0, control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(0, control.motion.y.target_velocity_q);
    ASSERT_EQ_I32(
        current_before_expiry - 100000,
        control.motion.x.current_velocity_q);
    ASSERT_TRUE(control.motion.x.current_velocity_q != 0);
    ASSERT_TRUE(control.velocity_target_present);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, -500, 250, 700U));
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_U32(700U, control.last_velocity_setpoint_ms);
    ASSERT_EQ_I32(
        -(int32_t)(500U * NAVMIN_VELOCITY_SCALE),
        control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(
        (int32_t)(250U * NAVMIN_VELOCITY_SCALE),
        control.motion.y.target_velocity_q);
}

static void test_dynamic_watchdog_timeout_preserves_setpoint_age(void)
{
    driver_sink_t sink_decrease;
    navmin_control_t decrease = make_control(&sink_decrease);
    driver_sink_t sink_increase;
    navmin_control_t increase = make_control(&sink_increase);

    configure_and_motor_on(&decrease, 1000U, 50000U, 1000U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&decrease, 500, 0, 100U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&decrease, 1000U, 1000U, 50000U, 50000U, 500U));
    ASSERT_EQ_U32(100U, decrease.last_velocity_setpoint_ms);
    ASSERT_TRUE(decrease.velocity_watchdog_armed);
    navmin_control_tick(&decrease, 800U);
    ASSERT_TRUE(!decrease.velocity_watchdog_armed);
    ASSERT_EQ_I32(0, decrease.motion.x.target_velocity_q);

    configure_and_motor_on(&increase, 1000U, 50000U, 500U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&increase, 500, 0, 100U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&increase, 1000U, 1000U, 50000U, 50000U, 1000U));
    ASSERT_EQ_U32(100U, increase.last_velocity_setpoint_ms);
    ASSERT_TRUE(increase.velocity_watchdog_armed);
    navmin_control_tick(&increase, 800U);
    ASSERT_TRUE(increase.velocity_watchdog_armed);
    ASSERT_TRUE(increase.motion.x.target_velocity_q != 0);
    navmin_control_tick(&increase, 1100U);
    ASSERT_TRUE(!increase.velocity_watchdog_armed);
    ASSERT_EQ_I32(0, increase.motion.x.target_velocity_q);
}

static void test_velocity_watchdog_uint32_wrap(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    configure_and_motor_on(&control, 1000U, 50000U, 200U);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        set_velocity_direct(&control, 500, 0, UINT32_MAX - 100U));

    navmin_control_tick(&control, 50U);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_TRUE(control.motion.x.target_velocity_q != 0);

    navmin_control_tick(&control, 99U);
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_EQ_I32(0, control.motion.x.target_velocity_q);
}

static void test_motor_off_and_emergency_disarm_velocity_watchdog(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);

    configure_and_motor_on(&control, 1000U, 50000U, 500U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, 500, 0, 100U));
    ASSERT_TRUE(control.velocity_watchdog_armed);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U, 0U));
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_TRUE(!control.velocity_target_present);
    ASSERT_EQ_I32(0, control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(0, control.motion.x.current_velocity_q);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_TRUE(!control.velocity_target_present);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, -400, 200, 200U));
    ASSERT_TRUE(control.velocity_watchdog_armed);
    navmin_control_emergency_stop(&control);
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_TRUE(!control.velocity_target_present);
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_EQ_I32(0, control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(0, control.motion.x.target_velocity_q);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
}

static void test_protocol_set_velocity_retry_timestamp_and_fresh_request(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    navmin_protocol_t protocol;
    response_log_t log = {0};
    executor_probe_t probe;
    navmin_protocol_executor_t executor;
    uint8_t config_payload[20];
    uint8_t velocity_payload[8];
    uint8_t request[28];
    size_t length;

    probe.control = &control;
    probe.execute_count = 0U;
    probe.emergency_count = 0U;
    executor.context = &probe;
    executor.emergency_stop = probe_emergency;
    executor.execute_command = probe_execute;
    navmin_protocol_init(&protocol, 0U, executor);

    build_config_payload(config_payload, 1000U, 1000U, 50000U, 50000U, 500U);
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_CONFIG,
        config_payload,
        (uint8_t)sizeof(config_payload));
    feed_frame_ending_at(&protocol, &log, request, length, 50U);
    assert_response_result(&log, 0U, NAVMIN_RESULT_OK);

    length = build_request(request, 1U, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U);
    feed_frame_ending_at(&protocol, &log, request, length, 70U);
    assert_response_result(&log, 1U, NAVMIN_RESULT_OK);

    build_velocity_payload(velocity_payload, 500, -250);
    length = build_request(
        request,
        2U,
        NAVMIN_COMMAND_SET_VELOCITY,
        velocity_payload,
        (uint8_t)sizeof(velocity_payload));
    feed_frame_ending_at(&protocol, &log, request, length, 100U);
    assert_response_result(&log, 2U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(3U, probe.execute_count);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_U32(100U, control.last_velocity_setpoint_ms);

    feed_frame_ending_at(&protocol, &log, request, length, 900U);
    assert_response_result(&log, 3U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(3U, probe.execute_count);
    ASSERT_EQ_U32(100U, control.last_velocity_setpoint_ms);
    ASSERT_TRUE(memcmp(log.bytes[2], log.bytes[3], NAVMIN_MIN_RESPONSE_LENGTH) == 0);

    length = build_request(
        request,
        3U,
        NAVMIN_COMMAND_SET_VELOCITY,
        velocity_payload,
        (uint8_t)sizeof(velocity_payload));
    feed_frame_ending_at(&protocol, &log, request, length, 1000U);
    assert_response_result(&log, 4U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(4U, probe.execute_count);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_U32(1000U, control.last_velocity_setpoint_ms);
}


static void test_move_relative_validation_bounds_and_safe_target_addition(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    uint8_t payload[8];
    int64_t accepted_target_x;
    int64_t accepted_target_y;
    int32_t invalid_values[] = {
        INT32_C(100001),
        -INT32_C(100001),
        INT32_MAX,
        INT32_MIN,
    };
    size_t i;

    build_relative_payload(payload, 1, 2);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_PARSE_ERROR,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOVE_RELATIVE, payload, 7U, 0U));

    /* Argument validation precedes config/motor state validation. */
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INVALID_ARGUMENT,
        move_relative_direct(&control, INT32_C(100001), 0));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_NOT_CONFIGURED,
        move_relative_direct(&control, 1, 0));

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 50000U, 50000U, 500U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_MOTORS_OFF,
        move_relative_direct(&control, 1, 0));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));

    control.motion.x.commanded_position_steps = INT64_C(123);
    control.motion.y.commanded_position_steps = -INT64_C(456);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        move_relative_direct(
            &control,
            NAVMIN_MAX_RELATIVE_DELTA_X_STEPS,
            -NAVMIN_MAX_RELATIVE_DELTA_Y_STEPS));
    ASSERT_EQ_I64(INT64_C(100123), control.relative_target_x_steps);
    ASSERT_EQ_I64(-INT64_C(100456), control.relative_target_y_steps);
    ASSERT_TRUE(control.relative_target_present);
    ASSERT_TRUE(!control.velocity_target_present);
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    accepted_target_x = control.relative_target_x_steps;
    accepted_target_y = control.relative_target_y_steps;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        move_relative_direct(
            &control,
            -NAVMIN_MAX_RELATIVE_DELTA_X_STEPS,
            NAVMIN_MAX_RELATIVE_DELTA_Y_STEPS));
    ASSERT_EQ_I64(-INT64_C(99877), control.relative_target_x_steps);
    ASSERT_EQ_I64(INT64_C(99544), control.relative_target_y_steps);
    accepted_target_x = control.relative_target_x_steps;
    accepted_target_y = control.relative_target_y_steps;

    for (i = 0U; i < sizeof(invalid_values) / sizeof(invalid_values[0]); ++i) {
        ASSERT_EQ_U32(
            NAVMIN_RESULT_INVALID_ARGUMENT,
            move_relative_direct(&control, invalid_values[i], 0));
        ASSERT_EQ_U32(
            NAVMIN_RESULT_INVALID_ARGUMENT,
            move_relative_direct(&control, 0, invalid_values[i]));
        ASSERT_EQ_I64(accepted_target_x, control.relative_target_x_steps);
        ASSERT_EQ_I64(accepted_target_y, control.relative_target_y_steps);
    }

    control.motion.x.commanded_position_steps = INT64_MAX - INT64_C(50);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INTERNAL_ERROR,
        move_relative_direct(&control, 100, 0));
    ASSERT_EQ_I64(accepted_target_x, control.relative_target_x_steps);
    ASSERT_EQ_I64(accepted_target_y, control.relative_target_y_steps);

    control.motion.x.commanded_position_steps = INT64_MIN + INT64_C(50);
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INTERNAL_ERROR,
        move_relative_direct(&control, -100, 0));
    ASSERT_EQ_I64(accepted_target_x, control.relative_target_x_steps);
    ASSERT_EQ_I64(accepted_target_y, control.relative_target_y_steps);
}

static void test_relative_planner_exact_axes_short_and_zero_moves(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned x_negative_before;
    unsigned y_positive_before;
    unsigned steps_before;

    configure_and_motor_on(&control, 1000U, 50000U, 500U);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 25, -17));
    navmin_control_tick(&control, 0U);
    ASSERT_EQ_I32(INT32_C(50000), control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(-INT32_C(50000), control.motion.y.current_velocity_q);
    ASSERT_EQ_I32(
        (int32_t)(1000U * NAVMIN_VELOCITY_SCALE),
        control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(
        -(int32_t)(1000U * NAVMIN_VELOCITY_SCALE),
        control.motion.y.target_velocity_q);
    run_until_relative_settled(&control, 200000U);
    assert_relative_axis_settled(&control.motion.x, 25);
    assert_relative_axis_settled(&control.motion.y, -17);
    ASSERT_EQ_U32(25U, sink.x_positive_step_count);
    ASSERT_EQ_U32(0U, sink.x_negative_step_count);
    ASSERT_EQ_U32(17U, sink.y_negative_step_count);
    ASSERT_EQ_U32(0U, sink.y_positive_step_count);

    x_negative_before = sink.x_negative_step_count;
    y_positive_before = sink.y_positive_step_count;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, -12, 9));
    run_until_relative_settled(&control, 200000U);
    assert_relative_axis_settled(&control.motion.x, 13);
    assert_relative_axis_settled(&control.motion.y, -8);
    ASSERT_EQ_U32(x_negative_before + 12U, sink.x_negative_step_count);
    ASSERT_EQ_U32(y_positive_before + 9U, sink.y_positive_step_count);

    steps_before = sink.step_count;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 0, 0));
    ASSERT_TRUE(control.relative_target_present);
    run_until_relative_settled(&control, 4U);
    ASSERT_EQ_U32(steps_before, sink.step_count);
    assert_relative_axis_settled(&control.motion.x, 13);
    assert_relative_axis_settled(&control.motion.y, -8);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 1, 0));
    run_until_relative_settled(&control, 10000U);
    assert_relative_axis_settled(&control.motion.x, 14);
    assert_relative_axis_settled(&control.motion.y, -8);
}

static void test_relative_replacement_uses_current_position_and_reverses_without_hard_stop(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned tick;
    int64_t replacement_base;
    int64_t reversal_base;
    int32_t velocity_before;
    int32_t phase_before;
    int64_t zero_delta_target;

    configure_and_motor_on(&control, 1000U, 100000U, 500U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 1000, 0));

    for (tick = 0U; tick < 300U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    ASSERT_TRUE(control.motion.x.commanded_position_steps > 0);
    ASSERT_TRUE(control.motion.x.current_velocity_q > 0);

    replacement_base = control.motion.x.commanded_position_steps;
    velocity_before = control.motion.x.current_velocity_q;
    phase_before = control.motion.x.step_phase_q;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 100, 0));
    ASSERT_EQ_I64(replacement_base + 100, control.relative_target_x_steps);
    ASSERT_EQ_I32(velocity_before, control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(phase_before, control.motion.x.step_phase_q);

    for (tick = 0U; tick < 40U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    ASSERT_TRUE(control.motion.x.current_velocity_q > 0);

    /* A zero-delta replacement targets the current position but preserves the
       already moving numeric state; the planner must brake and return if it
       cannot stop before that acceptance position. */
    zero_delta_target = control.motion.x.commanded_position_steps;
    velocity_before = control.motion.x.current_velocity_q;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 0, 0));
    ASSERT_EQ_I64(zero_delta_target, control.relative_target_x_steps);
    ASSERT_EQ_I32(velocity_before, control.motion.x.current_velocity_q);
    run_until_relative_settled(&control, 200000U);
    assert_relative_axis_settled(&control.motion.x, zero_delta_target);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 400, 0));
    for (tick = 0U; tick < 250U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    ASSERT_TRUE(control.motion.x.current_velocity_q > 0);
    reversal_base = control.motion.x.commanded_position_steps;
    velocity_before = control.motion.x.current_velocity_q;
    phase_before = control.motion.x.step_phase_q;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, -60, 0));
    ASSERT_EQ_I64(reversal_base - 60, control.relative_target_x_steps);
    ASSERT_EQ_I32(velocity_before, control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(phase_before, control.motion.x.step_phase_q);
    run_until_relative_settled(&control, 300000U);
    assert_relative_axis_settled(&control.motion.x, reversal_base - 60);
}

static void test_relative_and_velocity_behaviour_replace_each_other_without_watchdog_leak(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned tick;
    int32_t velocity_before;

    configure_and_motor_on(&control, 1000U, 50000U, 100U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, 500, 0, 10U));
    for (tick = 0U; tick < 100U; ++tick) {
        navmin_control_tick(&control, 10U);
    }
    ASSERT_TRUE(control.motion.x.current_velocity_q > 0);
    ASSERT_TRUE(control.velocity_watchdog_armed);

    velocity_before = control.motion.x.current_velocity_q;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 30, 0));
    ASSERT_TRUE(control.relative_target_present);
    ASSERT_TRUE(!control.velocity_target_present);
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    ASSERT_EQ_I32(velocity_before, control.motion.x.current_velocity_q);

    /* Relative motion has no communication watchdog even at a timestamp far
       beyond the previous velocity timeout. */
    navmin_control_tick(&control, 100000U);
    ASSERT_TRUE(control.relative_target_present);
    ASSERT_TRUE(!control.velocity_watchdog_armed);

    velocity_before = control.motion.x.current_velocity_q;
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, set_velocity_direct(&control, -300, 0, 200000U));
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_TRUE(control.velocity_target_present);
    ASSERT_TRUE(control.velocity_watchdog_armed);
    ASSERT_EQ_U32(200000U, control.last_velocity_setpoint_ms);
    ASSERT_EQ_I32(velocity_before, control.motion.x.current_velocity_q);
}

static void test_relative_planner_uses_dynamic_config_limits(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned tick;
    int64_t target_position;
    int32_t current_before;

    configure_and_motor_on(&control, 1000U, 100000U, 500U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 1000, 0));
    target_position = control.relative_target_x_steps;

    for (tick = 0U; tick < 180U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    ASSERT_TRUE(control.motion.x.current_velocity_q >
                (int32_t)(300U * NAVMIN_VELOCITY_SCALE));

    current_before = control.motion.x.current_velocity_q;
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 300U, 1000U, 100000U, 100000U, 500U));
    ASSERT_EQ_I64(target_position, control.relative_target_x_steps);
    ASSERT_EQ_I32(current_before, control.motion.x.current_velocity_q);
    ASSERT_EQ_I32(
        (int32_t)(300U * NAVMIN_VELOCITY_SCALE),
        control.motion.x.target_velocity_q);
    navmin_control_tick(&control, 0U);
    ASSERT_EQ_I32(current_before - INT32_C(100000), control.motion.x.current_velocity_q);

    current_before = control.motion.x.current_velocity_q;
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 20000U, 100000U, 500U));
    ASSERT_EQ_I64(target_position, control.relative_target_x_steps);
    ASSERT_EQ_I32(current_before, control.motion.x.current_velocity_q);
    navmin_control_tick(&control, 0U);
    ASSERT_EQ_I32(
        (int32_t)(1000U * NAVMIN_VELOCITY_SCALE),
        control.motion.x.target_velocity_q);
    ASSERT_EQ_I32(current_before + INT32_C(20000), control.motion.x.current_velocity_q);
    ASSERT_TRUE(!control.velocity_watchdog_armed);

    run_until_relative_settled(&control, 500000U);
    assert_relative_axis_settled(&control.motion.x, target_position);
}

static void test_relative_motor_off_emergency_and_motor_on_do_not_restore_target(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    unsigned tick;
    int64_t position_before_stop;

    configure_and_motor_on(&control, 1000U, 100000U, 500U);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, 200, 0));
    for (tick = 0U; tick < 200U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    position_before_stop = control.motion.x.commanded_position_steps;
    ASSERT_TRUE(control.relative_target_present);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U, 0U));
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    assert_numeric_motion_hard_stopped(&control, position_before_stop, 0);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_TRUE(!sink.enabled);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U, 0U));
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_EQ_I64(position_before_stop, control.motion.x.commanded_position_steps);
    ASSERT_EQ_I32(0, control.motion.x.current_velocity_q);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, move_relative_direct(&control, -200, 0));
    for (tick = 0U; tick < 200U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    position_before_stop = control.motion.x.commanded_position_steps;
    ASSERT_TRUE(control.relative_target_present);
    navmin_control_emergency_stop(&control);
    ASSERT_TRUE(!control.relative_target_present);
    ASSERT_TRUE(!control.velocity_watchdog_armed);
    assert_numeric_motion_hard_stopped(&control, position_before_stop, 0);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
}

static void test_protocol_move_relative_exact_retry_does_not_rebase_but_fresh_request_does(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    navmin_protocol_t protocol;
    response_log_t log = {0};
    executor_probe_t probe;
    navmin_protocol_executor_t executor;
    uint8_t config_payload[20];
    uint8_t relative_payload[8];
    uint8_t request[28];
    size_t length;
    unsigned tick;
    int64_t original_target;
    int64_t current_position;

    probe.control = &control;
    probe.execute_count = 0U;
    probe.emergency_count = 0U;
    executor.context = &probe;
    executor.emergency_stop = probe_emergency;
    executor.execute_command = probe_execute;
    navmin_protocol_init(&protocol, 0U, executor);

    build_config_payload(config_payload, 1000U, 1000U, 100000U, 100000U, 500U);
    length = build_request(
        request, 0U, NAVMIN_COMMAND_SET_CONFIG, config_payload, 20U);
    feed_frame_ending_at(&protocol, &log, request, length, 10U);
    assert_response_result(&log, 0U, NAVMIN_RESULT_OK);

    length = build_request(request, 1U, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U);
    feed_frame_ending_at(&protocol, &log, request, length, 30U);
    assert_response_result(&log, 1U, NAVMIN_RESULT_OK);

    build_relative_payload(relative_payload, 50, 0);
    length = build_request(
        request, 2U, NAVMIN_COMMAND_MOVE_RELATIVE, relative_payload, 8U);
    feed_frame_ending_at(&protocol, &log, request, length, 50U);
    assert_response_result(&log, 2U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(3U, probe.execute_count);
    original_target = control.relative_target_x_steps;
    ASSERT_EQ_I64(50, original_target);

    for (tick = 0U; tick < 500U; ++tick) {
        navmin_control_tick(&control, 0U);
    }
    current_position = control.motion.x.commanded_position_steps;
    ASSERT_TRUE(current_position > 0);
    ASSERT_TRUE(current_position < original_target);

    feed_frame_ending_at(&protocol, &log, request, length, 900U);
    assert_response_result(&log, 3U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(3U, probe.execute_count);
    ASSERT_EQ_I64(original_target, control.relative_target_x_steps);
    ASSERT_TRUE(memcmp(log.bytes[2], log.bytes[3], NAVMIN_MIN_RESPONSE_LENGTH) == 0);

    length = build_request(
        request, 3U, NAVMIN_COMMAND_MOVE_RELATIVE, relative_payload, 8U);
    feed_frame_ending_at(&protocol, &log, request, length, 1000U);
    assert_response_result(&log, 4U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(4U, probe.execute_count);
    ASSERT_EQ_I64(
        control.motion.x.commanded_position_steps + 50,
        control.relative_target_x_steps);
}

static void test_deferred_baud_command_does_not_fake_success(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    uint8_t payload[4] = {0};

    ASSERT_EQ_U32(
        NAVMIN_RESULT_INTERNAL_ERROR,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_SET_BAUDRATE, payload, 4U, 0U));
}

static void test_protocol_integration_set_config_motor_on_motor_off(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    navmin_protocol_t protocol;
    response_log_t log = {0};
    uint8_t config_payload[20];
    uint8_t request[28];
    uint32_t now = 0U;
    size_t length;

    navmin_protocol_init(
        &protocol,
        0U,
        navmin_control_protocol_executor(&control));

    build_config_payload(config_payload, 1000U, 2000U, 3000U, 4000U, 500U);
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_CONFIG,
        config_payload,
        (uint8_t)sizeof(config_payload));
    feed_frame(&protocol, &log, request, length, &now);
    assert_response_result(&log, 0U, NAVMIN_RESULT_OK);
    ASSERT_TRUE(control.configured);

    length = build_request(request, 1U, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U);
    feed_frame(&protocol, &log, request, length, &now);
    assert_response_result(&log, 1U, NAVMIN_RESULT_OK);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);

    seed_motion_state(&control);
    length = build_request(request, 2U, NAVMIN_COMMAND_EMERGENCY_STOP, NULL, 0U);
    feed_frame(&protocol, &log, request, length, &now);
    assert_response_result(&log, 2U, NAVMIN_RESULT_OK);
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);

    seed_motion_state(&control);
    length = build_request(request, 3U, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U);
    feed_frame(&protocol, &log, request, length, &now);
    assert_response_result(&log, 3U, NAVMIN_RESULT_OK);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_TRUE(!sink.enabled);
    assert_motion_cleared(&control);
    assert_numeric_motion_hard_stopped(&control, 321, -654);
}

static void test_protocol_exact_retries_do_not_repeat_control_side_effects(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    navmin_protocol_t protocol;
    response_log_t log = {0};
    executor_probe_t probe;
    navmin_protocol_executor_t executor;
    uint8_t config_payload[20];
    uint8_t request[28];
    uint32_t now = 0U;
    size_t length;

    probe.control = &control;
    probe.execute_count = 0U;
    probe.emergency_count = 0U;
    executor.context = &probe;
    executor.emergency_stop = probe_emergency;
    executor.execute_command = probe_execute;
    navmin_protocol_init(&protocol, 0U, executor);

    build_config_payload(config_payload, 1000U, 1000U, 5000U, 5000U, 200U);
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_CONFIG,
        config_payload,
        (uint8_t)sizeof(config_payload));
    feed_frame(&protocol, &log, request, length, &now);
    feed_frame(&protocol, &log, request, length, &now);
    ASSERT_EQ_U32(1U, probe.execute_count);
    ASSERT_EQ_U32(1U, sink.critical_enter_count);
    ASSERT_EQ_U32(1U, sink.critical_exit_count);
    ASSERT_TRUE(memcmp(log.bytes[0], log.bytes[1], NAVMIN_MIN_RESPONSE_LENGTH) == 0);

    length = build_request(request, 1U, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U);
    feed_frame(&protocol, &log, request, length, &now);
    feed_frame(&protocol, &log, request, length, &now);
    ASSERT_EQ_U32(2U, probe.execute_count);
    ASSERT_EQ_U32(1U, sink.enable_count);
    ASSERT_TRUE(memcmp(log.bytes[2], log.bytes[3], NAVMIN_MIN_RESPONSE_LENGTH) == 0);
}

int main(void)
{
    test_boot_is_unconfigured_and_motors_off();
    test_set_config_decodes_exact_little_endian_values();
    test_set_config_rejects_each_zero_field();
    test_set_config_accepts_all_firmware_bounds();
    test_set_config_rejects_each_bound_plus_one();
    test_invalid_config_keeps_previous_snapshot_and_valid_replaces_all();
    test_set_config_uses_one_critical_replacement_boundary();
    test_runtime_max_speed_decrease_and_increase_wire_motion_limits();
    test_runtime_acceleration_change_applies_on_next_motion_tick();
    test_control_owned_motion_routes_step_to_hardware_boundary();
    test_set_config_is_allowed_while_motors_on();
    test_motor_on_requires_config_then_is_idempotent();
    test_motor_off_before_config_and_while_off_is_idempotent();
    test_motor_off_while_on_hard_stops_and_disables();
    test_motor_off_then_motor_on_does_not_restore_numeric_motion();
    test_emergency_while_on_hard_stops_without_disabling_drivers();
    test_emergency_before_config_while_off_is_repeatable_without_latch();
    test_set_velocity_validation_clamp_and_extremes();
    test_set_velocity_zero_is_ordinary_setpoint();
    test_control_tick_watchdog_boundary_and_rearm();
    test_dynamic_watchdog_timeout_preserves_setpoint_age();
    test_velocity_watchdog_uint32_wrap();
    test_motor_off_and_emergency_disarm_velocity_watchdog();
    test_protocol_set_velocity_retry_timestamp_and_fresh_request();
    test_move_relative_validation_bounds_and_safe_target_addition();
    test_relative_planner_exact_axes_short_and_zero_moves();
    test_relative_replacement_uses_current_position_and_reverses_without_hard_stop();
    test_relative_and_velocity_behaviour_replace_each_other_without_watchdog_leak();
    test_relative_planner_uses_dynamic_config_limits();
    test_relative_motor_off_emergency_and_motor_on_do_not_restore_target();
    test_protocol_move_relative_exact_retry_does_not_rebase_but_fresh_request_does();
    test_deferred_baud_command_does_not_fake_success();
    test_protocol_integration_set_config_motor_on_motor_off();
    test_protocol_exact_retries_do_not_repeat_control_side_effects();

    puts("navmin firmware control host tests: PASS (34 cases)");
    return EXIT_SUCCESS;
}
