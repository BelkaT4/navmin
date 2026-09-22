#ifndef NAVMIN_STM32_HAL_H
#define NAVMIN_STM32_HAL_H

#include "main.h"
#include "navmin_stm32_app.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAVMIN_STM32_RX_RING_CAPACITY UINT16_C(256)
#define NAVMIN_STM32_STEP_PULSE_WIDTH_US UINT32_C(2)
#define NAVMIN_STM32_DIR_SETUP_US UINT32_C(1)
#define NAVMIN_STM32_UART_TX_TIMEOUT_MS UINT32_C(100)

typedef struct {
    UART_HandleTypeDef *uart;
    volatile uint16_t rx_head;
    volatile uint16_t rx_tail;
    uint8_t rx_bytes[NAVMIN_STM32_RX_RING_CAPACITY];
    uint32_t rx_timestamps_ms[NAVMIN_STM32_RX_RING_CAPACITY];
    uint8_t rx_it_byte;
    volatile bool rx_armed;
    volatile uint32_t rx_overflow_count;
    uint32_t current_baudrate;
    uint32_t saved_primask;
    uint32_t critical_depth;
} navmin_stm32_hal_t;

bool navmin_stm32_hal_init(navmin_stm32_hal_t *hardware, UART_HandleTypeDef *uart);

navmin_control_hardware_t navmin_stm32_hal_control_hardware(
    navmin_stm32_hal_t *hardware
);

navmin_stm32_transport_t navmin_stm32_hal_transport(
    navmin_stm32_hal_t *hardware
);

bool navmin_stm32_hal_start_rx(navmin_stm32_hal_t *hardware);
void navmin_stm32_hal_service_rx(navmin_stm32_hal_t *hardware);
void navmin_stm32_hal_rx_complete_isr(navmin_stm32_hal_t *hardware);
void navmin_stm32_hal_uart_error_isr(navmin_stm32_hal_t *hardware);

bool navmin_stm32_hal_pop_rx(
    navmin_stm32_hal_t *hardware,
    uint8_t *byte,
    uint32_t *timestamp_ms
);

#ifdef __cplusplus
}
#endif

#endif
