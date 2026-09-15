#ifndef NAVMIN_PROTOCOL_H
#define NAVMIN_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAVMIN_START_BYTE_0 UINT8_C(0xAA)
#define NAVMIN_START_BYTE_1 UINT8_C(0x55)
#define NAVMIN_MIN_REQUEST_LENGTH UINT8_C(8)
#define NAVMIN_MIN_RESPONSE_LENGTH UINT8_C(9)
#define NAVMIN_MAX_FRAME_LENGTH UINT16_C(255)
#define NAVMIN_MAX_REQUEST_PAYLOAD_LENGTH UINT16_C(247)
#define NAVMIN_RX_INTER_BYTE_TIMEOUT_MS UINT32_C(20)

typedef enum {
    NAVMIN_COMMAND_PING = 0x02,
    NAVMIN_COMMAND_SET_CONFIG = 0x10,
    NAVMIN_COMMAND_SET_BAUDRATE = 0x11,
    NAVMIN_COMMAND_MOVE_RELATIVE = 0x20,
    NAVMIN_COMMAND_SET_VELOCITY = 0x21,
    NAVMIN_COMMAND_EMERGENCY_STOP = 0x30,
    NAVMIN_COMMAND_MOTOR_ON = 0x31,
    NAVMIN_COMMAND_MOTOR_OFF = 0x32,
} navmin_command_code_t;

typedef enum {
    NAVMIN_RESULT_OK = 0x00,
    NAVMIN_RESULT_UNKNOWN_COMMAND = 0x01,
    NAVMIN_RESULT_PARSE_ERROR = 0x02,
    NAVMIN_RESULT_INVALID_ARGUMENT = 0x03,
    NAVMIN_RESULT_INVALID_REQUEST_ID = 0x04,
    NAVMIN_RESULT_MOTORS_OFF = 0x05,
    NAVMIN_RESULT_NOT_CONFIGURED = 0x06,
    NAVMIN_RESULT_INVALID_STATE = 0x07,
    NAVMIN_RESULT_INTERNAL_ERROR = 0x08,
} navmin_result_code_t;

typedef void (*navmin_emergency_stop_fn)(void *context);
typedef navmin_result_code_t (*navmin_execute_command_fn)(
    void *context,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
);
typedef void (*navmin_response_sink_fn)(
    void *context,
    const uint8_t *response,
    uint8_t response_length
);

typedef struct {
    void *context;
    navmin_emergency_stop_fn emergency_stop;
    navmin_execute_command_fn execute_command;
} navmin_protocol_executor_t;

typedef struct {
    uint8_t bytes[NAVMIN_MAX_FRAME_LENGTH];
    uint16_t length;
    uint32_t last_byte_time_ms;
    bool has_last_byte_time;
} navmin_request_parser_t;

typedef struct {
    bool valid;
    uint16_t request_id;
    uint8_t command_code;
    uint8_t payload_length;
    uint8_t payload[NAVMIN_MAX_REQUEST_PAYLOAD_LENGTH];
    uint8_t response[NAVMIN_MIN_RESPONSE_LENGTH];
    uint8_t response_length;
} navmin_retry_cache_t;

typedef struct {
    navmin_request_parser_t parser;
    uint16_t expected_request_id;
    navmin_retry_cache_t retry_cache;
    navmin_protocol_executor_t executor;
} navmin_protocol_t;

uint16_t navmin_crc16_modbus(const uint8_t *data, size_t length);

void navmin_protocol_init(
    navmin_protocol_t *protocol,
    uint16_t initial_expected_request_id,
    navmin_protocol_executor_t executor
);

void navmin_protocol_feed_byte(
    navmin_protocol_t *protocol,
    uint8_t byte,
    uint32_t now_ms,
    navmin_response_sink_fn response_sink,
    void *response_context
);

void navmin_protocol_poll(
    navmin_protocol_t *protocol,
    uint32_t now_ms,
    navmin_response_sink_fn response_sink,
    void *response_context
);

uint16_t navmin_protocol_expected_request_id(const navmin_protocol_t *protocol);

#ifdef __cplusplus
}
#endif

#endif
