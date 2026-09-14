#include "navmin_protocol.h"

#include <string.h>

#define NAVMIN_REQUEST_HEADER_LENGTH UINT8_C(6)
#define NAVMIN_CRC_LENGTH UINT8_C(2)
#define NAVMIN_RESPONSE_RESULT_OFFSET UINT8_C(6)

static uint16_t read_u16_le(const uint8_t *bytes)
{
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

static void write_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t)(value & UINT16_C(0x00FF));
    bytes[1] = (uint8_t)(value >> 8);
}

uint16_t navmin_crc16_modbus(const uint8_t *data, size_t length)
{
    uint16_t crc = UINT16_C(0xFFFF);
    size_t i;

    for (i = 0U; i < length; ++i) {
        unsigned bit;

        crc ^= data[i];
        for (bit = 0U; bit < 8U; ++bit) {
            if ((crc & UINT16_C(1)) != 0U) {
                crc = (uint16_t)((crc >> 1) ^ UINT16_C(0xA001));
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

static uint8_t expected_payload_length(uint8_t command_code, bool *known)
{
    *known = true;
    switch (command_code) {
    case NAVMIN_COMMAND_PING:
    case NAVMIN_COMMAND_EMERGENCY_STOP:
    case NAVMIN_COMMAND_MOTOR_ON:
    case NAVMIN_COMMAND_MOTOR_OFF:
        return 0U;
    case NAVMIN_COMMAND_SET_CONFIG:
        return 20U;
    case NAVMIN_COMMAND_SET_BAUDRATE:
        return 4U;
    case NAVMIN_COMMAND_MOVE_RELATIVE:
    case NAVMIN_COMMAND_SET_VELOCITY:
        return 8U;
    default:
        *known = false;
        return 0U;
    }
}

static uint8_t encode_response(
    uint16_t request_id,
    uint8_t command_code,
    navmin_result_code_t result,
    uint8_t output[NAVMIN_MIN_RESPONSE_LENGTH]
)
{
    uint16_t crc;

    output[0] = NAVMIN_START_BYTE_0;
    output[1] = NAVMIN_START_BYTE_1;
    output[2] = NAVMIN_MIN_RESPONSE_LENGTH;
    write_u16_le(&output[3], request_id);
    output[5] = command_code;
    output[NAVMIN_RESPONSE_RESULT_OFFSET] = (uint8_t)result;
    crc = navmin_crc16_modbus(&output[2], NAVMIN_MIN_RESPONSE_LENGTH - 4U);
    write_u16_le(&output[7], crc);
    return NAVMIN_MIN_RESPONSE_LENGTH;
}

static bool cache_signature_matches(
    const navmin_retry_cache_t *cache,
    uint16_t request_id,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    if (!cache->valid || cache->request_id != request_id ||
        cache->command_code != command_code ||
        cache->payload_length != payload_length) {
        return false;
    }
    return payload_length == 0U ||
           memcmp(cache->payload, payload, payload_length) == 0;
}

static void cache_transaction(
    navmin_retry_cache_t *cache,
    uint16_t request_id,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    const uint8_t *response,
    uint8_t response_length
)
{
    cache->valid = true;
    cache->request_id = request_id;
    cache->command_code = command_code;
    cache->payload_length = payload_length;
    if (payload_length > 0U) {
        memcpy(cache->payload, payload, payload_length);
    }
    memcpy(cache->response, response, response_length);
    cache->response_length = response_length;
}

static void emit_response(
    navmin_response_sink_fn response_sink,
    void *response_context,
    const uint8_t *response,
    uint8_t response_length
)
{
    if (response_sink != NULL) {
        response_sink(response_context, response, response_length);
    }
}

static void process_valid_request(
    navmin_protocol_t *protocol,
    const uint8_t *frame,
    uint8_t frame_length,
    navmin_response_sink_fn response_sink,
    void *response_context
)
{
    uint16_t request_id = read_u16_le(&frame[3]);
    uint8_t command_code = frame[5];
    uint8_t payload_length = (uint8_t)(frame_length - NAVMIN_MIN_REQUEST_LENGTH);
    const uint8_t *payload = &frame[NAVMIN_REQUEST_HEADER_LENGTH];
    uint8_t response[NAVMIN_MIN_RESPONSE_LENGTH];
    uint8_t response_length;
    navmin_result_code_t result;
    bool known;
    uint8_t command_payload_length;

    if (cache_signature_matches(
            &protocol->retry_cache,
            request_id,
            command_code,
            payload,
            payload_length)) {
        emit_response(
            response_sink,
            response_context,
            protocol->retry_cache.response,
            protocol->retry_cache.response_length);
        return;
    }

    if (command_code == NAVMIN_COMMAND_EMERGENCY_STOP) {
        if (protocol->executor.emergency_stop != NULL) {
            protocol->executor.emergency_stop(protocol->executor.context);
        }
        protocol->expected_request_id = (uint16_t)(request_id + UINT16_C(1));
        result = payload_length == 0U ? NAVMIN_RESULT_OK : NAVMIN_RESULT_PARSE_ERROR;
        response_length = encode_response(request_id, command_code, result, response);
        cache_transaction(
            &protocol->retry_cache,
            request_id,
            command_code,
            payload,
            payload_length,
            response,
            response_length);
        emit_response(response_sink, response_context, response, response_length);
        return;
    }

    /* For ordinary requests, the last cached ID may only be reused as an
       exact retry. Emergency is handled above as the sequence resync boundary. */
    if (protocol->retry_cache.valid &&
        protocol->retry_cache.request_id == request_id) {
        response_length = encode_response(
            request_id,
            command_code,
            NAVMIN_RESULT_INVALID_REQUEST_ID,
            response);
        emit_response(response_sink, response_context, response, response_length);
        return;
    }

    if (request_id != protocol->expected_request_id) {
        response_length = encode_response(
            request_id,
            command_code,
            NAVMIN_RESULT_INVALID_REQUEST_ID,
            response);
        emit_response(response_sink, response_context, response, response_length);
        return;
    }

    command_payload_length = expected_payload_length(command_code, &known);
    if (!known) {
        result = NAVMIN_RESULT_UNKNOWN_COMMAND;
    } else if (payload_length != command_payload_length) {
        result = NAVMIN_RESULT_PARSE_ERROR;
    } else if (command_code == NAVMIN_COMMAND_PING) {
        result = NAVMIN_RESULT_OK;
    } else if (protocol->executor.execute_command == NULL) {
        result = NAVMIN_RESULT_INTERNAL_ERROR;
    } else {
        result = protocol->executor.execute_command(
            protocol->executor.context,
            command_code,
            payload,
            payload_length);
        if ((uint8_t)result > (uint8_t)NAVMIN_RESULT_INTERNAL_ERROR) {
            result = NAVMIN_RESULT_INTERNAL_ERROR;
        }
    }

    protocol->expected_request_id = (uint16_t)(request_id + UINT16_C(1));
    response_length = encode_response(request_id, command_code, result, response);
    cache_transaction(
        &protocol->retry_cache,
        request_id,
        command_code,
        payload,
        payload_length,
        response,
        response_length);
    emit_response(response_sink, response_context, response, response_length);
}

static void parser_drop_prefix(navmin_request_parser_t *parser, uint16_t count)
{
    if (count >= parser->length) {
        parser->length = 0U;
        return;
    }
    memmove(parser->bytes, &parser->bytes[count], parser->length - count);
    parser->length = (uint16_t)(parser->length - count);
}

static bool parser_starts_with_start(const navmin_request_parser_t *parser)
{
    return parser->length >= 2U &&
           parser->bytes[0] == NAVMIN_START_BYTE_0 &&
           parser->bytes[1] == NAVMIN_START_BYTE_1;
}

static bool parser_has_incomplete_candidate(const navmin_request_parser_t *parser)
{
    uint8_t frame_length;

    if (!parser_starts_with_start(parser)) {
        return false;
    }
    if (parser->length < 3U) {
        return true;
    }
    frame_length = parser->bytes[2];
    return frame_length >= NAVMIN_MIN_REQUEST_LENGTH && parser->length < frame_length;
}

static void parser_find_start(navmin_request_parser_t *parser)
{
    uint16_t i;

    if (parser_starts_with_start(parser)) {
        return;
    }

    for (i = 0U; i + 1U < parser->length; ++i) {
        if (parser->bytes[i] == NAVMIN_START_BYTE_0 &&
            parser->bytes[i + 1U] == NAVMIN_START_BYTE_1) {
            parser_drop_prefix(parser, i);
            return;
        }
    }

    if (parser->length > 0U &&
        parser->bytes[parser->length - 1U] == NAVMIN_START_BYTE_0) {
        parser->bytes[0] = NAVMIN_START_BYTE_0;
        parser->length = 1U;
    } else {
        parser->length = 0U;
    }
}

static void parser_process_buffer(
    navmin_protocol_t *protocol,
    navmin_response_sink_fn response_sink,
    void *response_context
)
{
    navmin_request_parser_t *parser = &protocol->parser;

    for (;;) {
        uint8_t frame_length;
        uint16_t expected_crc;
        uint16_t actual_crc;

        parser_find_start(parser);
        if (parser->length < 3U) {
            return;
        }

        frame_length = parser->bytes[2];
        if (frame_length < NAVMIN_MIN_REQUEST_LENGTH) {
            parser_drop_prefix(parser, 1U);
            continue;
        }
        if (parser->length < frame_length) {
            return;
        }

        expected_crc = read_u16_le(&parser->bytes[frame_length - NAVMIN_CRC_LENGTH]);
        actual_crc = navmin_crc16_modbus(
            &parser->bytes[2],
            (size_t)frame_length - 4U);
        if (expected_crc != actual_crc) {
            parser_drop_prefix(parser, 1U);
            continue;
        }

        process_valid_request(
            protocol,
            parser->bytes,
            frame_length,
            response_sink,
            response_context);
        parser_drop_prefix(parser, frame_length);
    }
}

static void parser_apply_timeout(
    navmin_protocol_t *protocol,
    uint32_t now_ms,
    navmin_response_sink_fn response_sink,
    void *response_context
)
{
    navmin_request_parser_t *parser = &protocol->parser;
    uint32_t elapsed;

    if (!parser->has_last_byte_time || !parser_has_incomplete_candidate(parser)) {
        return;
    }

    elapsed = now_ms - parser->last_byte_time_ms;
    if (elapsed <= NAVMIN_RX_INTER_BYTE_TIMEOUT_MS) {
        return;
    }

    /* Every incomplete candidate reconstructed only from the stale buffered
       bytes has the same expired inter-byte gap. Keep resynchronizing until
       the buffer is no longer waiting on such a candidate. */
    do {
        parser_drop_prefix(parser, 1U);
        parser_process_buffer(protocol, response_sink, response_context);
    } while (parser_has_incomplete_candidate(parser));
}

void navmin_protocol_init(
    navmin_protocol_t *protocol,
    uint16_t initial_expected_request_id,
    navmin_protocol_executor_t executor
)
{
    memset(protocol, 0, sizeof(*protocol));
    protocol->expected_request_id = initial_expected_request_id;
    protocol->executor = executor;
}

void navmin_protocol_feed_byte(
    navmin_protocol_t *protocol,
    uint8_t byte,
    uint32_t now_ms,
    navmin_response_sink_fn response_sink,
    void *response_context
)
{
    navmin_request_parser_t *parser = &protocol->parser;

    parser_apply_timeout(protocol, now_ms, response_sink, response_context);
    if (parser->length >= NAVMIN_MAX_FRAME_LENGTH) {
        parser_drop_prefix(parser, 1U);
        parser_process_buffer(protocol, response_sink, response_context);
    }

    parser->bytes[parser->length] = byte;
    ++parser->length;
    parser->last_byte_time_ms = now_ms;
    parser->has_last_byte_time = true;
    parser_process_buffer(protocol, response_sink, response_context);
}

void navmin_protocol_poll(
    navmin_protocol_t *protocol,
    uint32_t now_ms,
    navmin_response_sink_fn response_sink,
    void *response_context
)
{
    parser_apply_timeout(protocol, now_ms, response_sink, response_context);
}

uint16_t navmin_protocol_expected_request_id(const navmin_protocol_t *protocol)
{
    return protocol->expected_request_id;
}
