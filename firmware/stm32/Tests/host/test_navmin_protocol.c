#include "navmin_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LENGTH(a) (sizeof(a) / sizeof((a)[0]))
#define TEST_RESPONSE_CAPACITY 8U

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
    uint8_t bytes[TEST_RESPONSE_CAPACITY][NAVMIN_MIN_RESPONSE_LENGTH];
    uint8_t lengths[TEST_RESPONSE_CAPACITY];
    unsigned count;
    unsigned safety_count_at_emit[TEST_RESPONSE_CAPACITY];
    unsigned *safety_count;
} response_log_t;

typedef struct {
    unsigned emergency_count;
    unsigned execute_count;
    uint8_t last_command;
    uint32_t last_execute_time_ms;
    navmin_result_code_t next_result;
} executor_state_t;

static uint16_t read_u16_le(const uint8_t *bytes)
{
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

static void write_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t)(value & 0xFFU);
    bytes[1] = (uint8_t)(value >> 8);
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

static void emergency_stop(void *context)
{
    executor_state_t *state = context;
    ++state->emergency_count;
}

static navmin_result_code_t execute_command(
    void *context,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
)
{
    executor_state_t *state = context;
    (void)payload;
    (void)payload_length;
    ++state->execute_count;
    state->last_command = command_code;
    state->last_execute_time_ms = now_ms;
    return state->next_result;
}

static void capture_response(
    void *context,
    const uint8_t *response,
    uint8_t response_length
)
{
    response_log_t *log = context;

    ASSERT_TRUE(log->count < TEST_RESPONSE_CAPACITY);
    ASSERT_EQ_U32(NAVMIN_MIN_RESPONSE_LENGTH, response_length);
    memcpy(log->bytes[log->count], response, response_length);
    log->lengths[log->count] = response_length;
    log->safety_count_at_emit[log->count] =
        log->safety_count == NULL ? 0U : *log->safety_count;
    ++log->count;
}

static void init_protocol(
    navmin_protocol_t *protocol,
    executor_state_t *state,
    uint16_t initial_expected_request_id
)
{
    navmin_protocol_executor_t executor;

    memset(state, 0, sizeof(*state));
    state->next_result = NAVMIN_RESULT_OK;
    executor.context = state;
    executor.emergency_stop = emergency_stop;
    executor.execute_command = execute_command;
    navmin_protocol_init(protocol, initial_expected_request_id, executor);
}

static void feed_bytes(
    navmin_protocol_t *protocol,
    response_log_t *log,
    const uint8_t *bytes,
    size_t length,
    uint32_t *now_ms
)
{
    size_t i;

    for (i = 0U; i < length; ++i) {
        navmin_protocol_feed_byte(
            protocol,
            bytes[i],
            *now_ms,
            capture_response,
            log);
        ++*now_ms;
    }
}

static void assert_response(
    const response_log_t *log,
    unsigned index,
    uint16_t request_id,
    uint8_t command,
    navmin_result_code_t result
)
{
    const uint8_t *response;
    uint16_t expected_crc;
    uint16_t actual_crc;

    ASSERT_TRUE(index < log->count);
    response = log->bytes[index];
    ASSERT_EQ_U32(NAVMIN_START_BYTE_0, response[0]);
    ASSERT_EQ_U32(NAVMIN_START_BYTE_1, response[1]);
    ASSERT_EQ_U32(NAVMIN_MIN_RESPONSE_LENGTH, response[2]);
    ASSERT_EQ_U32(request_id, read_u16_le(&response[3]));
    ASSERT_EQ_U32(command, response[5]);
    ASSERT_EQ_U32(result, response[6]);
    expected_crc = read_u16_le(&response[7]);
    actual_crc = navmin_crc16_modbus(&response[2], NAVMIN_MIN_RESPONSE_LENGTH - 4U);
    ASSERT_EQ_U32(expected_crc, actual_crc);
}

static void test_crc_known_vectors(void)
{
    static const uint8_t vector[] = "123456789";
    static const uint8_t ping_crc_input[] = {0x08, 0x34, 0x12, 0x02};

    ASSERT_EQ_U32(0x4B37, navmin_crc16_modbus(vector, 9U));
    ASSERT_EQ_U32(0xEBCE, navmin_crc16_modbus(ping_crc_input, ARRAY_LENGTH(ping_crc_input)));
}

static void test_command_and_result_vocabulary_matches_v1(void)
{
    ASSERT_EQ_U32(0x02U, NAVMIN_COMMAND_PING);
    ASSERT_EQ_U32(0x10U, NAVMIN_COMMAND_SET_CONFIG);
    ASSERT_EQ_U32(0x11U, NAVMIN_COMMAND_SET_BAUDRATE);
    ASSERT_EQ_U32(0x20U, NAVMIN_COMMAND_MOVE_RELATIVE);
    ASSERT_EQ_U32(0x21U, NAVMIN_COMMAND_SET_VELOCITY);
    ASSERT_EQ_U32(0x30U, NAVMIN_COMMAND_EMERGENCY_STOP);
    ASSERT_EQ_U32(0x31U, NAVMIN_COMMAND_MOTOR_ON);
    ASSERT_EQ_U32(0x32U, NAVMIN_COMMAND_MOTOR_OFF);

    ASSERT_EQ_U32(0x00U, NAVMIN_RESULT_OK);
    ASSERT_EQ_U32(0x01U, NAVMIN_RESULT_UNKNOWN_COMMAND);
    ASSERT_EQ_U32(0x02U, NAVMIN_RESULT_PARSE_ERROR);
    ASSERT_EQ_U32(0x03U, NAVMIN_RESULT_INVALID_ARGUMENT);
    ASSERT_EQ_U32(0x04U, NAVMIN_RESULT_INVALID_REQUEST_ID);
    ASSERT_EQ_U32(0x05U, NAVMIN_RESULT_MOTORS_OFF);
    ASSERT_EQ_U32(0x06U, NAVMIN_RESULT_NOT_CONFIGURED);
    ASSERT_EQ_U32(0x07U, NAVMIN_RESULT_INVALID_STATE);
    ASSERT_EQ_U32(0x08U, NAVMIN_RESULT_INTERNAL_ERROR);
}

static void test_pc_ping_wire_vector_is_accepted(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint32_t now = 0U;
    static const uint8_t pc_request[] = {
        0xAA, 0x55, 0x08, 0x34, 0x12, 0x02, 0xCE, 0xEB
    };

    init_protocol(&protocol, &state, 0x1234U);
    feed_bytes(&protocol, &log, pc_request, sizeof(pc_request), &now);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0x1234U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_minimal_valid_frame_and_response_encoding(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;
    static const uint8_t expected_response[] = {
        0xAA, 0x55, 0x09, 0x34, 0x12, 0x02, 0x00, 0x57, 0x54
    };

    init_protocol(&protocol, &state, 0x1234U);
    build_request(request, 0x1234U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(1U, log.count);
    ASSERT_TRUE(memcmp(log.bytes[0], expected_response, sizeof(expected_response)) == 0);
    ASSERT_EQ_U32(0x1235U, navmin_protocol_expected_request_id(&protocol));
}

static void test_garbage_before_start(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;
    static const uint8_t garbage[] = {0x00, 0x55, 0x10, 0xAA, 0x00, 0x7E};

    init_protocol(&protocol, &state, 0U);
    build_request(request, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, garbage, sizeof(garbage), &now);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_overlapping_start_prefix(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;
    uint8_t extra_aa = NAVMIN_START_BYTE_0;

    init_protocol(&protocol, &state, 0U);
    build_request(request, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, &extra_aa, 1U, &now);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_short_length_resynchronizes(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;
    static const uint8_t short_candidate[] = {0xAA, 0x55, 0x07};

    init_protocol(&protocol, &state, 0U);
    build_request(request, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, short_candidate, sizeof(short_candidate), &now);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_incomplete_frame_timeout_reuses_accumulated_bytes(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t nested[8];
    uint8_t partial[12];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    build_request(nested, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    partial[0] = 0xAA;
    partial[1] = 0x55;
    partial[2] = 20U;
    partial[3] = 0x99;
    memcpy(&partial[4], nested, sizeof(nested));
    feed_bytes(&protocol, &log, partial, sizeof(partial), &now);
    ASSERT_EQ_U32(0U, log.count);

    navmin_protocol_poll(
        &protocol,
        (now - 1U) + NAVMIN_RX_INTER_BYTE_TIMEOUT_MS,
        capture_response,
        &log);
    ASSERT_EQ_U32(0U, log.count);

    navmin_protocol_poll(
        &protocol,
        (now - 1U) + NAVMIN_RX_INTER_BYTE_TIMEOUT_MS + 1U,
        capture_response,
        &log);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_crc_failure_has_no_response_and_resynchronizes(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t nested[8];
    uint8_t bad_outer[16];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    build_request(nested, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    memset(bad_outer, 0x33, sizeof(bad_outer));
    bad_outer[0] = 0xAA;
    bad_outer[1] = 0x55;
    bad_outer[2] = (uint8_t)sizeof(bad_outer);
    bad_outer[3] = 0x77;
    memcpy(&bad_outer[4], nested, sizeof(nested));
    bad_outer[14] = 0x00;
    bad_outer[15] = 0x00;

    feed_bytes(&protocol, &log, bad_outer, sizeof(bad_outer), &now);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_start_bytes_inside_payload_do_not_restart_frame(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[16];
    uint32_t now = 0U;
    static const uint8_t payload[8] = {0xAA, 0x55, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44};

    init_protocol(&protocol, &state, 0U);
    build_request(
        request,
        0U,
        NAVMIN_COMMAND_MOVE_RELATIVE,
        payload,
        (uint8_t)sizeof(payload));
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(1U, log.count);
    ASSERT_EQ_U32(1U, state.execute_count);
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_MOVE_RELATIVE, NAVMIN_RESULT_OK);
}

static void test_initial_expected_id(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;

    init_protocol(&protocol, &state, 37U);

    ASSERT_EQ_U32(37U, navmin_protocol_expected_request_id(&protocol));
}

static void test_normal_id_increment(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t first[8];
    uint8_t second[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 37U);
    build_request(first, 37U, NAVMIN_COMMAND_PING, NULL, 0U);
    build_request(second, 38U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, first, sizeof(first), &now);
    ASSERT_EQ_U32(38U, navmin_protocol_expected_request_id(&protocol));
    feed_bytes(&protocol, &log, second, sizeof(second), &now);
    ASSERT_EQ_U32(39U, navmin_protocol_expected_request_id(&protocol));
    ASSERT_EQ_U32(2U, log.count);
}


static void test_maximum_length_frame_is_accepted(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[NAVMIN_MAX_FRAME_LENGTH];
    uint8_t payload[NAVMIN_MAX_REQUEST_PAYLOAD_LENGTH];
    uint32_t now = 0U;

    memset(payload, 0x5AU, sizeof(payload));
    init_protocol(&protocol, &state, 0U);
    ASSERT_EQ_U32(
        NAVMIN_MAX_FRAME_LENGTH,
        build_request(
            request,
            0U,
            0x99U,
            payload,
            (uint8_t)sizeof(payload)));
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(1U, log.count);
    assert_response(&log, 0U, 0U, 0x99U, NAVMIN_RESULT_UNKNOWN_COMMAND);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));
}

static void test_request_id_wraps(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0xFFFFU);
    build_request(request, 0xFFFFU, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(0U, navmin_protocol_expected_request_id(&protocol));
    assert_response(&log, 0U, 0xFFFFU, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_wrong_ordinary_id_does_not_advance(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 5U);
    build_request(request, 7U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(5U, navmin_protocol_expected_request_id(&protocol));
    assert_response(
        &log,
        0U,
        7U,
        NAVMIN_COMMAND_PING,
        NAVMIN_RESULT_INVALID_REQUEST_ID);
}

static void test_ordinary_error_consumes_correct_id(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t malformed[9];
    uint8_t next[8];
    uint32_t now = 0U;
    uint8_t junk_payload = 0x42U;

    init_protocol(&protocol, &state, 0U);
    build_request(malformed, 0U, NAVMIN_COMMAND_PING, &junk_payload, 1U);
    build_request(next, 1U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, malformed, sizeof(malformed), &now);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));
    assert_response(&log, 0U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_PARSE_ERROR);

    feed_bytes(&protocol, &log, next, sizeof(next), &now);
    ASSERT_EQ_U32(2U, navmin_protocol_expected_request_id(&protocol));
    assert_response(&log, 1U, 1U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_exact_retry_returns_cached_response_without_second_execution(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t request[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    state.next_result = NAVMIN_RESULT_OK;
    build_request(request, 0U, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U);
    feed_bytes(&protocol, &log, request, sizeof(request), &now);
    ASSERT_EQ_U32(1U, state.execute_count);
    ASSERT_EQ_U32(7U, state.last_execute_time_ms);

    state.next_result = NAVMIN_RESULT_INTERNAL_ERROR;
    feed_bytes(&protocol, &log, request, sizeof(request), &now);

    ASSERT_EQ_U32(2U, log.count);
    ASSERT_EQ_U32(1U, state.execute_count);
    ASSERT_EQ_U32(7U, state.last_execute_time_ms);
    ASSERT_TRUE(memcmp(log.bytes[0], log.bytes[1], NAVMIN_MIN_RESPONSE_LENGTH) == 0);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));
}

static void test_same_cached_id_different_signature_is_invalid(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t first[8];
    uint8_t different[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    build_request(first, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    build_request(different, 0U, NAVMIN_COMMAND_MOTOR_OFF, NULL, 0U);
    feed_bytes(&protocol, &log, first, sizeof(first), &now);
    feed_bytes(&protocol, &log, different, sizeof(different), &now);

    ASSERT_EQ_U32(2U, log.count);
    ASSERT_EQ_U32(0U, state.execute_count);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));
    assert_response(
        &log,
        1U,
        0U,
        NAVMIN_COMMAND_MOTOR_OFF,
        NAVMIN_RESULT_INVALID_REQUEST_ID);
}

static void test_emergency_with_cached_ordinary_id_is_special_resync(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t ping[8];
    uint8_t emergency[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    build_request(ping, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    build_request(emergency, 0U, NAVMIN_COMMAND_EMERGENCY_STOP, NULL, 0U);

    feed_bytes(&protocol, &log, ping, sizeof(ping), &now);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));

    feed_bytes(&protocol, &log, emergency, sizeof(emergency), &now);

    ASSERT_EQ_U32(2U, log.count);
    ASSERT_EQ_U32(1U, state.emergency_count);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));
    assert_response(
        &log,
        1U,
        0U,
        NAVMIN_COMMAND_EMERGENCY_STOP,
        NAVMIN_RESULT_OK);
    ASSERT_TRUE(protocol.retry_cache.valid);
    ASSERT_EQ_U32(0U, protocol.retry_cache.request_id);
    ASSERT_EQ_U32(NAVMIN_COMMAND_EMERGENCY_STOP, protocol.retry_cache.command_code);
    ASSERT_EQ_U32(0U, protocol.retry_cache.payload_length);
    ASSERT_EQ_U32(NAVMIN_MIN_RESPONSE_LENGTH, protocol.retry_cache.response_length);
    ASSERT_TRUE(
        memcmp(
            protocol.retry_cache.response,
            log.bytes[1],
            NAVMIN_MIN_RESPONSE_LENGTH) == 0);
}

static void test_emergency_is_accepted_with_unexpected_id(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t emergency[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 3U);
    build_request(emergency, 900U, NAVMIN_COMMAND_EMERGENCY_STOP, NULL, 0U);
    feed_bytes(&protocol, &log, emergency, sizeof(emergency), &now);

    ASSERT_EQ_U32(1U, state.emergency_count);
    assert_response(&log, 0U, 900U, NAVMIN_COMMAND_EMERGENCY_STOP, NAVMIN_RESULT_OK);
}

static void test_emergency_resets_expected_id_to_n_plus_one(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t emergency[8];
    uint8_t next[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    build_request(emergency, 0xFFFFU, NAVMIN_COMMAND_EMERGENCY_STOP, NULL, 0U);
    build_request(next, 0U, NAVMIN_COMMAND_PING, NULL, 0U);
    feed_bytes(&protocol, &log, emergency, sizeof(emergency), &now);
    ASSERT_EQ_U32(0U, navmin_protocol_expected_request_id(&protocol));
    feed_bytes(&protocol, &log, next, sizeof(next), &now);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&protocol));
    assert_response(&log, 1U, 0U, NAVMIN_COMMAND_PING, NAVMIN_RESULT_OK);
}

static void test_malformed_emergency_runs_safety_before_parse_error(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t emergency[9];
    uint8_t malformed_payload = 0x99U;
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    log.safety_count = &state.emergency_count;
    build_request(
        emergency,
        44U,
        NAVMIN_COMMAND_EMERGENCY_STOP,
        &malformed_payload,
        1U);
    feed_bytes(&protocol, &log, emergency, sizeof(emergency), &now);

    ASSERT_EQ_U32(1U, state.emergency_count);
    ASSERT_EQ_U32(1U, log.safety_count_at_emit[0]);
    ASSERT_EQ_U32(45U, navmin_protocol_expected_request_id(&protocol));
    assert_response(
        &log,
        0U,
        44U,
        NAVMIN_COMMAND_EMERGENCY_STOP,
        NAVMIN_RESULT_PARSE_ERROR);
}

static void test_exact_emergency_retry_does_not_run_safety_twice(void)
{
    navmin_protocol_t protocol;
    executor_state_t state;
    response_log_t log = {0};
    uint8_t emergency[8];
    uint32_t now = 0U;

    init_protocol(&protocol, &state, 0U);
    build_request(emergency, 12U, NAVMIN_COMMAND_EMERGENCY_STOP, NULL, 0U);
    feed_bytes(&protocol, &log, emergency, sizeof(emergency), &now);
    feed_bytes(&protocol, &log, emergency, sizeof(emergency), &now);

    ASSERT_EQ_U32(2U, log.count);
    ASSERT_EQ_U32(1U, state.emergency_count);
    ASSERT_TRUE(memcmp(log.bytes[0], log.bytes[1], NAVMIN_MIN_RESPONSE_LENGTH) == 0);
    ASSERT_EQ_U32(13U, navmin_protocol_expected_request_id(&protocol));
}

int main(void)
{
    test_crc_known_vectors();
    test_command_and_result_vocabulary_matches_v1();
    test_pc_ping_wire_vector_is_accepted();
    test_minimal_valid_frame_and_response_encoding();
    test_garbage_before_start();
    test_overlapping_start_prefix();
    test_short_length_resynchronizes();
    test_incomplete_frame_timeout_reuses_accumulated_bytes();
    test_crc_failure_has_no_response_and_resynchronizes();
    test_start_bytes_inside_payload_do_not_restart_frame();
    test_initial_expected_id();
    test_normal_id_increment();
    test_maximum_length_frame_is_accepted();
    test_request_id_wraps();
    test_wrong_ordinary_id_does_not_advance();
    test_ordinary_error_consumes_correct_id();
    test_exact_retry_returns_cached_response_without_second_execution();
    test_same_cached_id_different_signature_is_invalid();
    test_emergency_with_cached_ordinary_id_is_special_resync();
    test_emergency_is_accepted_with_unexpected_id();
    test_emergency_resets_expected_id_to_n_plus_one();
    test_malformed_emergency_runs_safety_before_parse_error();
    test_exact_emergency_retry_does_not_run_safety_twice();

    puts("navmin firmware protocol host tests: PASS (23 cases)");
    return EXIT_SUCCESS;
}
