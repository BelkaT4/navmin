#include "navmin_stm32_app.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(condition)                                                   \
    do {                                                                         \
        if (!(condition)) {                                                       \
            fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n",               \
                    __FILE__, __LINE__, #condition);                              \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define ASSERT_EQ_U32(expected, actual)                                           \
    do {                                                                         \
        uint32_t expected_value = (uint32_t)(expected);                            \
        uint32_t actual_value = (uint32_t)(actual);                                \
        if (expected_value != actual_value) {                                      \
            fprintf(stderr,                                                       \
                    "ASSERT_EQ_U32 failed at %s:%d: expected %lu, got %lu\n",   \
                    __FILE__, __LINE__,                                            \
                    (unsigned long)expected_value,                                 \
                    (unsigned long)actual_value);                                  \
            exit(EXIT_FAILURE);                                                    \
        }                                                                         \
    } while (0)

typedef struct {
    uint32_t baudrate;
    uint32_t tx_baudrate[16];
    uint8_t tx_bytes[16][NAVMIN_MIN_RESPONSE_LENGTH];
    uint8_t tx_length[16];
    uint32_t tx_count;
    uint32_t apply_count;
    uint32_t apply_after_tx_count;
    bool fail_next_tx;
} fake_transport_t;

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

static size_t build_request(
    uint8_t *output,
    uint16_t request_id,
    uint8_t command,
    const uint8_t *payload,
    uint8_t payload_length
)
{
    size_t length = (size_t)NAVMIN_MIN_REQUEST_LENGTH + payload_length;
    uint16_t crc;

    output[0] = NAVMIN_START_BYTE_0;
    output[1] = NAVMIN_START_BYTE_1;
    output[2] = (uint8_t)length;
    write_u16_le(&output[3], request_id);
    output[5] = command;
    if (payload_length > 0U) {
        memcpy(&output[6], payload, payload_length);
    }
    crc = navmin_crc16_modbus(&output[2], length - 4U);
    write_u16_le(&output[length - 2U], crc);
    return length;
}

static void feed_request(
    navmin_stm32_app_t *app,
    const uint8_t *request,
    size_t length,
    uint32_t now_ms
)
{
    size_t index;

    for (index = 0U; index < length; ++index) {
        navmin_stm32_app_feed_byte(app, request[index], now_ms);
    }
}

static bool fake_transmit_response(
    void *context,
    const uint8_t *response,
    uint8_t response_length
)
{
    fake_transport_t *fake = context;
    uint32_t index = fake->tx_count;

    ASSERT_TRUE(index < 16U);
    ASSERT_TRUE(response_length == NAVMIN_MIN_RESPONSE_LENGTH);
    fake->tx_baudrate[index] = fake->baudrate;
    fake->tx_length[index] = response_length;
    memcpy(fake->tx_bytes[index], response, response_length);
    ++fake->tx_count;

    if (fake->fail_next_tx) {
        fake->fail_next_tx = false;
        return false;
    }
    return true;
}

static bool fake_apply_baudrate(void *context, uint32_t baudrate)
{
    fake_transport_t *fake = context;

    ++fake->apply_count;
    fake->apply_after_tx_count = fake->tx_count;
    fake->baudrate = baudrate;
    return true;
}

static void init_app(navmin_stm32_app_t *app, fake_transport_t *fake)
{
    navmin_control_hardware_t control_hardware = {0};
    navmin_stm32_transport_t transport;

    memset(fake, 0, sizeof(*fake));
    fake->baudrate = NAVMIN_STM32_STARTUP_BAUDRATE;

    transport.context = fake;
    transport.transmit_response = fake_transmit_response;
    transport.apply_baudrate = fake_apply_baudrate;
    navmin_stm32_app_init(app, control_hardware, transport);
}

static navmin_result_code_t response_result(const fake_transport_t *fake, uint32_t index)
{
    ASSERT_TRUE(index < fake->tx_count);
    return (navmin_result_code_t)fake->tx_bytes[index][6];
}

static void test_set_baudrate_response_precedes_switch_and_exact_retry_is_cached(void)
{
    fake_transport_t fake;
    navmin_stm32_app_t app;
    init_app(&app, &fake);
    uint8_t payload[4];
    uint8_t request[12];
    size_t length;

    write_u32_le(payload, UINT32_C(115200));
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));

    feed_request(&app, request, length, 100U);

    ASSERT_EQ_U32(1U, fake.tx_count);
    ASSERT_EQ_U32(9600U, fake.tx_baudrate[0]);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, response_result(&fake, 0U));
    ASSERT_EQ_U32(1U, fake.apply_count);
    ASSERT_EQ_U32(1U, fake.apply_after_tx_count);
    ASSERT_EQ_U32(115200U, fake.baudrate);
    ASSERT_EQ_U32(115200U, app.current_baudrate);
    ASSERT_TRUE(!app.pending_baudrate_valid);

    feed_request(&app, request, length, 900U);

    ASSERT_EQ_U32(2U, fake.tx_count);
    ASSERT_EQ_U32(115200U, fake.tx_baudrate[1]);
    ASSERT_EQ_U32(1U, fake.apply_count);
    ASSERT_TRUE(memcmp(
        fake.tx_bytes[0],
        fake.tx_bytes[1],
        NAVMIN_MIN_RESPONSE_LENGTH) == 0);
    ASSERT_EQ_U32(1U, navmin_protocol_expected_request_id(&app.protocol));
}

static void test_tx_failure_keeps_old_baud_until_cached_retry_is_transmitted(void)
{
    fake_transport_t fake;
    navmin_stm32_app_t app;
    init_app(&app, &fake);
    uint8_t payload[4];
    uint8_t request[12];
    size_t length;

    fake.fail_next_tx = true;
    write_u32_le(payload, UINT32_C(57600));
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));

    feed_request(&app, request, length, 10U);
    ASSERT_EQ_U32(1U, fake.tx_count);
    ASSERT_EQ_U32(0U, fake.apply_count);
    ASSERT_EQ_U32(9600U, app.current_baudrate);
    ASSERT_TRUE(app.pending_baudrate_valid);

    feed_request(&app, request, length, 20U);
    ASSERT_EQ_U32(2U, fake.tx_count);
    ASSERT_EQ_U32(9600U, fake.tx_baudrate[1]);
    ASSERT_EQ_U32(1U, fake.apply_count);
    ASSERT_EQ_U32(57600U, app.current_baudrate);
    ASSERT_TRUE(!app.pending_baudrate_valid);
    ASSERT_TRUE(memcmp(
        fake.tx_bytes[0],
        fake.tx_bytes[1],
        NAVMIN_MIN_RESPONSE_LENGTH) == 0);
}


static void test_new_ordinary_transaction_cancels_unconfirmed_pending_baud_switch(void)
{
    fake_transport_t fake;
    navmin_stm32_app_t app;
    init_app(&app, &fake);
    uint8_t payload[4];
    uint8_t request[12];
    size_t length;

    fake.fail_next_tx = true;
    write_u32_le(payload, UINT32_C(57600));
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));
    feed_request(&app, request, length, 10U);
    ASSERT_TRUE(app.pending_baudrate_valid);
    ASSERT_EQ_U32(9600U, app.current_baudrate);

    length = build_request(
        request,
        1U,
        NAVMIN_COMMAND_PING,
        NULL,
        0U);
    feed_request(&app, request, length, 20U);

    ASSERT_EQ_U32(2U, fake.tx_count);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, response_result(&fake, 1U));
    ASSERT_EQ_U32(0U, fake.apply_count);
    ASSERT_EQ_U32(9600U, app.current_baudrate);
    ASSERT_TRUE(!app.pending_baudrate_valid);

    /* The accepted PING replaced the firmware one-entry retry cache. Retrying
       the old baud transaction can no longer apply the stale switch. */
    write_u32_le(payload, UINT32_C(57600));
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));
    feed_request(&app, request, length, 30U);
    ASSERT_EQ_U32(3U, fake.tx_count);
    ASSERT_EQ_U32(NAVMIN_RESULT_INVALID_REQUEST_ID, response_result(&fake, 2U));
    ASSERT_EQ_U32(0U, fake.apply_count);
    ASSERT_EQ_U32(9600U, app.current_baudrate);
}

static void test_new_emergency_cancels_unconfirmed_pending_baud_switch(void)
{
    fake_transport_t fake;
    navmin_stm32_app_t app;
    init_app(&app, &fake);
    uint8_t payload[4];
    uint8_t request[12];
    size_t length;

    fake.fail_next_tx = true;
    write_u32_le(payload, UINT32_C(57600));
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));
    feed_request(&app, request, length, 10U);
    ASSERT_TRUE(app.pending_baudrate_valid);
    ASSERT_EQ_U32(9600U, app.current_baudrate);

    length = build_request(
        request,
        42U,
        NAVMIN_COMMAND_EMERGENCY_STOP,
        NULL,
        0U);
    feed_request(&app, request, length, 20U);

    ASSERT_EQ_U32(2U, fake.tx_count);
    ASSERT_EQ_U32(NAVMIN_RESULT_OK, response_result(&fake, 1U));
    ASSERT_EQ_U32(0U, fake.apply_count);
    ASSERT_EQ_U32(9600U, app.current_baudrate);
    ASSERT_TRUE(!app.pending_baudrate_valid);
    ASSERT_EQ_U32(43U, navmin_protocol_expected_request_id(&app.protocol));
}

static void test_set_baudrate_validation_and_motors_off_requirement(void)
{
    fake_transport_t fake;
    navmin_stm32_app_t app;
    init_app(&app, &fake);
    uint8_t payload[4];
    uint8_t request[12];
    size_t length;

    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        NULL,
        0U);
    feed_request(&app, request, length, 1U);
    ASSERT_EQ_U32(NAVMIN_RESULT_PARSE_ERROR, response_result(&fake, 0U));
    ASSERT_EQ_U32(0U, fake.apply_count);

    write_u32_le(payload, UINT32_C(230400));
    length = build_request(
        request,
        1U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));
    feed_request(&app, request, length, 2U);
    ASSERT_EQ_U32(NAVMIN_RESULT_INVALID_ARGUMENT, response_result(&fake, 1U));
    ASSERT_EQ_U32(0U, fake.apply_count);

    app.control.motors_on = true;
    write_u32_le(payload, UINT32_C(115200));
    length = build_request(
        request,
        2U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));
    feed_request(&app, request, length, 3U);
    ASSERT_EQ_U32(NAVMIN_RESULT_INVALID_STATE, response_result(&fake, 2U));
    ASSERT_EQ_U32(0U, fake.apply_count);
    ASSERT_EQ_U32(9600U, app.current_baudrate);
    ASSERT_TRUE(!app.pending_baudrate_valid);
}

static void test_repeating_current_baud_is_ok_without_reconfiguration(void)
{
    fake_transport_t fake;
    navmin_stm32_app_t app;
    init_app(&app, &fake);
    uint8_t payload[4];
    uint8_t request[12];
    size_t length;

    write_u32_le(payload, NAVMIN_STM32_STARTUP_BAUDRATE);
    length = build_request(
        request,
        0U,
        NAVMIN_COMMAND_SET_BAUDRATE,
        payload,
        (uint8_t)sizeof(payload));
    feed_request(&app, request, length, 5U);

    ASSERT_EQ_U32(NAVMIN_RESULT_OK, response_result(&fake, 0U));
    ASSERT_EQ_U32(9600U, fake.tx_baudrate[0]);
    ASSERT_EQ_U32(0U, fake.apply_count);
    ASSERT_EQ_U32(9600U, app.current_baudrate);
}

int main(void)
{
    test_set_baudrate_response_precedes_switch_and_exact_retry_is_cached();
    test_tx_failure_keeps_old_baud_until_cached_retry_is_transmitted();
    test_new_ordinary_transaction_cancels_unconfirmed_pending_baud_switch();
    test_new_emergency_cancels_unconfirmed_pending_baud_switch();
    test_set_baudrate_validation_and_motors_off_requirement();
    test_repeating_current_baud_is_ok_without_reconfiguration();

    puts("navmin STM32 app host tests: PASS (6 cases)");
    return EXIT_SUCCESS;
}
