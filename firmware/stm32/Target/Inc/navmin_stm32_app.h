#ifndef NAVMIN_STM32_APP_H
#define NAVMIN_STM32_APP_H

#include "navmin_control.h"
#include "navmin_protocol.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAVMIN_STM32_STARTUP_BAUDRATE UINT32_C(9600)

typedef bool (*navmin_stm32_transmit_response_fn)(
    void *context,
    const uint8_t *response,
    uint8_t response_length
);
typedef bool (*navmin_stm32_apply_baudrate_fn)(void *context, uint32_t baudrate);

typedef struct {
    void *context;
    navmin_stm32_transmit_response_fn transmit_response;
    navmin_stm32_apply_baudrate_fn apply_baudrate;
} navmin_stm32_transport_t;

typedef struct {
    navmin_protocol_t protocol;
    navmin_control_t control;
    navmin_stm32_transport_t transport;
    uint32_t current_baudrate;
    uint32_t pending_baudrate;
    bool pending_baudrate_valid;
} navmin_stm32_app_t;

void navmin_stm32_app_init(
    navmin_stm32_app_t *app,
    navmin_control_hardware_t control_hardware,
    navmin_stm32_transport_t transport
);

void navmin_stm32_app_feed_byte(
    navmin_stm32_app_t *app,
    uint8_t byte,
    uint32_t now_ms
);

void navmin_stm32_app_poll(navmin_stm32_app_t *app, uint32_t now_ms);

void navmin_stm32_app_control_tick(navmin_stm32_app_t *app, uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif
