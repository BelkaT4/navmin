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

typedef struct {
    bool enabled;
    unsigned call_count;
    unsigned enable_count;
    unsigned disable_count;
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

static navmin_control_t make_control(driver_sink_t *sink)
{
    navmin_control_t control;
    navmin_control_hardware_t hardware;

    memset(sink, 0, sizeof(*sink));
    hardware.context = sink;
    hardware.set_drivers_enabled = set_drivers_enabled;
    navmin_control_init(&control, hardware);
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
        (uint8_t)sizeof(payload));
}

static void seed_motion_state(navmin_control_t *control)
{
    control->velocity_target_present = true;
    control->relative_target_present = true;
}

static void assert_motion_cleared(const navmin_control_t *control)
{
    ASSERT_TRUE(!control->velocity_target_present);
    ASSERT_TRUE(!control->relative_target_present);
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
    uint8_t payload_length
)
{
    executor_probe_t *probe = context;

    ++probe->execute_count;
    return navmin_control_execute_command(
        probe->control,
        command_code,
        payload,
        payload_length);
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

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 2000U, 3000U, 4000U, 500U));
    original = control.config;

    ASSERT_EQ_U32(
        NAVMIN_RESULT_INVALID_ARGUMENT,
        apply_config(&control, 0U, 2222U, 3333U, 4444U, 555U));
    ASSERT_TRUE(memcmp(&control.config, &original, sizeof(original)) == 0);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1111U, 2222U, 3333U, 4444U, 555U));
    ASSERT_EQ_U32(1111U, control.config.max_speed_x_steps_s);
    ASSERT_EQ_U32(2222U, control.config.max_speed_y_steps_s);
    ASSERT_EQ_U32(3333U, control.config.acceleration_x_steps_s2);
    ASSERT_EQ_U32(4444U, control.config.acceleration_y_steps_s2);
    ASSERT_EQ_U32(555U, control.config.velocity_watchdog_timeout_ms);
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
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
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
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
    ASSERT_TRUE(!control.motors_on);
    ASSERT_EQ_U32(0U, sink.enable_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
    ASSERT_TRUE(control.motors_on);
    ASSERT_TRUE(sink.enabled);
    ASSERT_EQ_U32(1U, sink.enable_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
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
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U));
    assert_motion_cleared(&control);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_EQ_U32(calls_after_init, sink.call_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U));
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
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
    seed_motion_state(&control);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U));

    ASSERT_TRUE(!control.motors_on);
    ASSERT_TRUE(!sink.enabled);
    ASSERT_EQ_U32(1U, sink.enable_count);
    ASSERT_EQ_U32(2U, sink.disable_count);
    assert_motion_cleared(&control);
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
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
    seed_motion_state(&control);
    calls_before = sink.call_count;

    navmin_control_emergency_stop(&control);

    assert_motion_cleared(&control);
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
    ASSERT_TRUE(!control.configured);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_EQ_U32(calls_after_init, sink.call_count);

    seed_motion_state(&control);
    navmin_control_emergency_stop(&control);
    assert_motion_cleared(&control);
    ASSERT_EQ_U32(calls_after_init, sink.call_count);

    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        apply_config(&control, 1000U, 1000U, 5000U, 5000U, 200U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_OK,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOTOR_ON, NULL, 0U));
    ASSERT_TRUE(control.motors_on);
}

static void test_unimplemented_motion_and_baud_commands_do_not_fake_success(void)
{
    driver_sink_t sink;
    navmin_control_t control = make_control(&sink);
    uint8_t payload[8] = {0};

    ASSERT_EQ_U32(
        NAVMIN_RESULT_INTERNAL_ERROR,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_MOVE_RELATIVE, payload, 8U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INTERNAL_ERROR,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_SET_VELOCITY, payload, 8U));
    ASSERT_EQ_U32(
        NAVMIN_RESULT_INTERNAL_ERROR,
        navmin_control_execute_command(
            &control, NAVMIN_COMMAND_SET_BAUDRATE, payload, 4U));
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

    seed_motion_state(&control);
    length = build_request(request, 3U, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U);
    feed_frame(&protocol, &log, request, length, &now);
    assert_response_result(&log, 3U, NAVMIN_RESULT_OK);
    ASSERT_TRUE(!control.motors_on);
    ASSERT_TRUE(!sink.enabled);
    assert_motion_cleared(&control);
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
    test_set_config_is_allowed_while_motors_on();
    test_motor_on_requires_config_then_is_idempotent();
    test_motor_off_before_config_and_while_off_is_idempotent();
    test_motor_off_while_on_hard_stops_and_disables();
    test_emergency_while_on_hard_stops_without_disabling_drivers();
    test_emergency_before_config_while_off_is_repeatable_without_latch();
    test_unimplemented_motion_and_baud_commands_do_not_fake_success();
    test_protocol_integration_set_config_motor_on_motor_off();
    test_protocol_exact_retries_do_not_repeat_control_side_effects();

    puts("navmin firmware control host tests: PASS (15 cases)");
    return EXIT_SUCCESS;
}
