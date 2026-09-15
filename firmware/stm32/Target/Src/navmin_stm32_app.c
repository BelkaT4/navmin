#include "navmin_stm32_app.h"

#include <string.h>

#define NAVMIN_SET_BAUDRATE_PAYLOAD_LENGTH UINT8_C(4)

static uint32_t read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static bool baudrate_is_supported(uint32_t baudrate)
{
    switch (baudrate) {
    case UINT32_C(9600):
    case UINT32_C(19200):
    case UINT32_C(38400):
    case UINT32_C(57600):
    case UINT32_C(115200):
        return true;
    default:
        return false;
    }
}

static navmin_result_code_t execute_set_baudrate(
    navmin_stm32_app_t *app,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    uint32_t baudrate;

    if (payload == NULL || payload_length != NAVMIN_SET_BAUDRATE_PAYLOAD_LENGTH) {
        return NAVMIN_RESULT_PARSE_ERROR;
    }

    baudrate = read_u32_le(payload);
    if (!baudrate_is_supported(baudrate)) {
        return NAVMIN_RESULT_INVALID_ARGUMENT;
    }
    if (app->control.motors_on) {
        return NAVMIN_RESULT_INVALID_STATE;
    }
    if (baudrate == app->current_baudrate) {
        return NAVMIN_RESULT_OK;
    }
    if (app->transport.apply_baudrate == NULL) {
        return NAVMIN_RESULT_INTERNAL_ERROR;
    }

    /* The protocol response sink transmits the OK response at the current
       (old) baud and only then consumes this pending switch. */
    app->pending_baudrate = baudrate;
    app->pending_baudrate_valid = true;
    return NAVMIN_RESULT_OK;
}

static navmin_result_code_t execute_command(
    void *context,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
)
{
    navmin_stm32_app_t *app = context;

    if (app == NULL) {
        return NAVMIN_RESULT_INTERNAL_ERROR;
    }

    /* Reaching the executor means this is a new transaction, not an exact
       retry (which the protocol cache handles earlier). Any older baud switch
       whose response never completed is therefore abandoned before this new
       transaction can publish a response. */
    app->pending_baudrate_valid = false;

    if (command_code == NAVMIN_COMMAND_SET_BAUDRATE) {
        return execute_set_baudrate(app, payload, payload_length);
    }
    return navmin_control_execute_command(
        &app->control,
        command_code,
        payload,
        payload_length,
        now_ms);
}

static void emergency_stop(void *context)
{
    navmin_stm32_app_t *app = context;

    if (app != NULL) {
        /* A new Emergency is also a sequence-resync boundary. Do not allow a
           stale, unconfirmed prior SET_BAUDRATE to switch UART after the
           Emergency response. Exact Emergency retry never re-enters here. */
        app->pending_baudrate_valid = false;
        navmin_control_emergency_stop(&app->control);
    }
}

static bool response_is_valid_minimum(
    const uint8_t *response,
    uint8_t response_length
)
{
    return response != NULL && response_length >= NAVMIN_MIN_RESPONSE_LENGTH;
}

static bool response_completes_pending_baud_switch(
    const uint8_t *response,
    uint8_t response_length
)
{
    return response_is_valid_minimum(response, response_length) &&
           response[5] == NAVMIN_COMMAND_SET_BAUDRATE &&
           response[6] == NAVMIN_RESULT_OK;
}

static bool response_replaces_pending_baud_transaction(
    const uint8_t *response,
    uint8_t response_length
)
{
    /* A non-INVALID_REQUEST_ID ordinary response means protocol sequencing has
       accepted a newer transaction and replaced the one-entry retry cache. A
       previously unconfirmed baud switch can no longer be exact-retried and
       must not remain armed. This also covers protocol-core-owned PING and
       payload/unknown-command errors that never enter execute_command(). */
    return response_is_valid_minimum(response, response_length) &&
           response[6] != NAVMIN_RESULT_INVALID_REQUEST_ID &&
           !response_completes_pending_baud_switch(response, response_length);
}

static void response_sink(
    void *context,
    const uint8_t *response,
    uint8_t response_length
)
{
    navmin_stm32_app_t *app = context;
    bool transmitted;

    if (app == NULL || app->transport.transmit_response == NULL) {
        return;
    }

    if (app->pending_baudrate_valid &&
        response_replaces_pending_baud_transaction(response, response_length)) {
        app->pending_baudrate_valid = false;
    }

    /* transmit_response() is required to return true only after the complete
       response has physically left USART. SET_BAUDRATE therefore cannot
       change BRR before the old-baud response is complete. */
    transmitted = app->transport.transmit_response(
        app->transport.context,
        response,
        response_length);
    if (!transmitted ||
        !app->pending_baudrate_valid ||
        !response_completes_pending_baud_switch(response, response_length)) {
        return;
    }

    if (app->transport.apply_baudrate(
            app->transport.context,
            app->pending_baudrate)) {
        app->current_baudrate = app->pending_baudrate;
        app->pending_baudrate_valid = false;
    }
}

void navmin_stm32_app_init(
    navmin_stm32_app_t *app,
    navmin_control_hardware_t control_hardware,
    navmin_stm32_transport_t transport
)
{
    navmin_protocol_executor_t executor;

    memset(app, 0, sizeof(*app));
    app->transport = transport;
    app->current_baudrate = NAVMIN_STM32_STARTUP_BAUDRATE;

    navmin_control_init(&app->control, control_hardware);

    executor.context = app;
    executor.emergency_stop = emergency_stop;
    executor.execute_command = execute_command;
    navmin_protocol_init(&app->protocol, 0U, executor);
}

void navmin_stm32_app_feed_byte(
    navmin_stm32_app_t *app,
    uint8_t byte,
    uint32_t now_ms
)
{
    if (app == NULL) {
        return;
    }
    navmin_protocol_feed_byte(
        &app->protocol,
        byte,
        now_ms,
        response_sink,
        app);
}

void navmin_stm32_app_poll(navmin_stm32_app_t *app, uint32_t now_ms)
{
    if (app == NULL) {
        return;
    }
    navmin_protocol_poll(&app->protocol, now_ms, response_sink, app);
}

void navmin_stm32_app_control_tick(navmin_stm32_app_t *app, uint32_t now_ms)
{
    if (app != NULL) {
        navmin_control_tick(&app->control, now_ms);
    }
}
