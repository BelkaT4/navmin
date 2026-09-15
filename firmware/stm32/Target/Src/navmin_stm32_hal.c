#include "navmin_stm32_hal.h"

#include <string.h>

#define NAVMIN_STM32_RX_RING_MASK (NAVMIN_STM32_RX_RING_CAPACITY - UINT16_C(1))
#define NAVMIN_X_STEP_PIN GPIO_PIN_7
#define NAVMIN_X_DIR_PIN GPIO_PIN_8
#define NAVMIN_X_ENABLE_PIN GPIO_PIN_9
#define NAVMIN_Y_STEP_PIN GPIO_PIN_4
#define NAVMIN_Y_DIR_PIN GPIO_PIN_5
#define NAVMIN_Y_ENABLE_PIN GPIO_PIN_6

_Static_assert(
    (NAVMIN_STM32_RX_RING_CAPACITY & NAVMIN_STM32_RX_RING_MASK) == 0U,
    "RX ring capacity must be a power of two");
_Static_assert(
    NAVMIN_MAX_FRAME_LENGTH <= NAVMIN_STM32_RX_RING_CAPACITY,
    "RX ring must hold at least one maximum protocol frame");

static void delay_us(uint32_t microseconds)
{
    uint32_t cycles_per_us = SystemCoreClock / UINT32_C(1000000);
    uint32_t wait_cycles = cycles_per_us * microseconds;
    uint32_t start = DWT->CYCCNT;

    while ((uint32_t)(DWT->CYCCNT - start) < wait_cycles) {
        __NOP();
    }
}

static GPIO_PinState direction_pin_state(navmin_step_direction_t direction)
{
    /* Internal +X is logical right and +Y is logical up after PC-side invert
       conversion. Both board axes use GPIO low for that positive direction. */
    return direction == NAVMIN_STEP_DIRECTION_POSITIVE ?
        GPIO_PIN_RESET : GPIO_PIN_SET;
}

static void set_drivers_enabled(void *context, bool enabled)
{
    navmin_stm32_hal_t *hardware = context;
    GPIO_PinState state = enabled ? GPIO_PIN_RESET : GPIO_PIN_SET;

    (void)hardware;
    HAL_GPIO_WritePin(
        GPIOB,
        NAVMIN_X_ENABLE_PIN | NAVMIN_Y_ENABLE_PIN,
        state);
}

static void emit_step(
    void *context,
    navmin_motion_axis_t axis,
    navmin_step_direction_t direction
)
{
    navmin_stm32_hal_t *hardware = context;
    uint16_t direction_pin;
    uint16_t step_pin;

    (void)hardware;
    if (axis == NAVMIN_MOTION_AXIS_X) {
        direction_pin = NAVMIN_X_DIR_PIN;
        step_pin = NAVMIN_X_STEP_PIN;
    } else if (axis == NAVMIN_MOTION_AXIS_Y) {
        direction_pin = NAVMIN_Y_DIR_PIN;
        step_pin = NAVMIN_Y_STEP_PIN;
    } else {
        return;
    }

    HAL_GPIO_WritePin(GPIOB, direction_pin, direction_pin_state(direction));
    delay_us(NAVMIN_STM32_DIR_SETUP_US);
    HAL_GPIO_WritePin(GPIOB, step_pin, GPIO_PIN_SET);
    delay_us(NAVMIN_STM32_STEP_PULSE_WIDTH_US);
    HAL_GPIO_WritePin(GPIOB, step_pin, GPIO_PIN_RESET);
}

static void enter_critical(void *context)
{
    navmin_stm32_hal_t *hardware = context;
    uint32_t primask = __get_PRIMASK();

    __disable_irq();
    if (hardware->critical_depth == 0U) {
        hardware->saved_primask = primask;
    }
    ++hardware->critical_depth;
    __DMB();
}

static void exit_critical(void *context)
{
    navmin_stm32_hal_t *hardware = context;

    __DMB();
    if (hardware->critical_depth == 0U) {
        return;
    }

    --hardware->critical_depth;
    if (hardware->critical_depth == 0U && hardware->saved_primask == 0U) {
        __enable_irq();
    }
}

static bool transmit_response(
    void *context,
    const uint8_t *response,
    uint8_t response_length
)
{
    navmin_stm32_hal_t *hardware = context;
    HAL_StatusTypeDef status;

    status = HAL_UART_Transmit(
        hardware->uart,
        (uint8_t *)response,
        response_length,
        NAVMIN_STM32_UART_TX_TIMEOUT_MS);
    if (status != HAL_OK) {
        return false;
    }

    /* STM32F1 HAL_UART_Transmit() normally waits for TC already. Keep an
       explicit bounded TC wait so SET_BAUDRATE never relies on a HAL-version
       detail that could report completion before the final stop bit left TX. */
    {
        uint32_t started_ms = HAL_GetTick();
        while (__HAL_UART_GET_FLAG(hardware->uart, UART_FLAG_TC) == RESET) {
            if ((uint32_t)(HAL_GetTick() - started_ms) >=
                NAVMIN_STM32_UART_TX_TIMEOUT_MS) {
                return false;
            }
        }
    }
    return true;
}

static void reset_rx_ring(navmin_stm32_hal_t *hardware)
{
    hardware->rx_head = 0U;
    hardware->rx_tail = 0U;
}

static bool apply_baudrate(void *context, uint32_t baudrate)
{
    navmin_stm32_hal_t *hardware = context;
    uint32_t previous_baudrate = hardware->current_baudrate;

    hardware->rx_armed = false;
    (void)HAL_UART_AbortReceive(hardware->uart);
    if (HAL_UART_DeInit(hardware->uart) != HAL_OK) {
        (void)navmin_stm32_hal_start_rx(hardware);
        return false;
    }

    reset_rx_ring(hardware);
    hardware->uart->Init.BaudRate = baudrate;
    if (HAL_UART_Init(hardware->uart) == HAL_OK) {
        hardware->current_baudrate = baudrate;
        if (navmin_stm32_hal_start_rx(hardware)) {
            return true;
        }
        (void)HAL_UART_DeInit(hardware->uart);
    }

    /* A valid protocol baud should not fail HAL init/rearm. If it does, restore
       the previous UART configuration so PC recovery can still probe old baud. */
    hardware->uart->Init.BaudRate = previous_baudrate;
    if (HAL_UART_Init(hardware->uart) == HAL_OK) {
        hardware->current_baudrate = previous_baudrate;
        (void)navmin_stm32_hal_start_rx(hardware);
    }
    return false;
}

bool navmin_stm32_hal_init(navmin_stm32_hal_t *hardware, UART_HandleTypeDef *uart)
{
    if (hardware == NULL || uart == NULL) {
        return false;
    }

    memset(hardware, 0, sizeof(*hardware));
    hardware->uart = uart;
    hardware->current_baudrate = NAVMIN_STM32_STARTUP_BAUDRATE;

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    if ((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) != 0U) {
        return false;
    }
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB();
    __ISB();
    return true;
}

navmin_control_hardware_t navmin_stm32_hal_control_hardware(
    navmin_stm32_hal_t *hardware
)
{
    navmin_control_hardware_t result;

    result.context = hardware;
    result.set_drivers_enabled = set_drivers_enabled;
    result.emit_step = emit_step;
    result.enter_critical = enter_critical;
    result.exit_critical = exit_critical;
    return result;
}

navmin_stm32_transport_t navmin_stm32_hal_transport(
    navmin_stm32_hal_t *hardware
)
{
    navmin_stm32_transport_t result;

    result.context = hardware;
    result.transmit_response = transmit_response;
    result.apply_baudrate = apply_baudrate;
    return result;
}

bool navmin_stm32_hal_start_rx(navmin_stm32_hal_t *hardware)
{
    if (hardware == NULL || hardware->uart == NULL) {
        return false;
    }
    if (hardware->rx_armed) {
        return true;
    }

    if (HAL_UART_Receive_IT(hardware->uart, &hardware->rx_it_byte, 1U) != HAL_OK) {
        return false;
    }
    hardware->rx_armed = true;
    return true;
}

void navmin_stm32_hal_service_rx(navmin_stm32_hal_t *hardware)
{
    if (hardware != NULL && !hardware->rx_armed) {
        (void)navmin_stm32_hal_start_rx(hardware);
    }
}

void navmin_stm32_hal_rx_complete_isr(navmin_stm32_hal_t *hardware)
{
    uint16_t head;
    uint16_t next;

    if (hardware == NULL) {
        return;
    }

    hardware->rx_armed = false;
    head = hardware->rx_head;
    next = (uint16_t)((head + UINT16_C(1)) & NAVMIN_STM32_RX_RING_MASK);
    if (next == hardware->rx_tail) {
        ++hardware->rx_overflow_count;
    } else {
        hardware->rx_bytes[head] = hardware->rx_it_byte;
        hardware->rx_timestamps_ms[head] = HAL_GetTick();
        __DMB();
        hardware->rx_head = next;
    }

    (void)navmin_stm32_hal_start_rx(hardware);
}

void navmin_stm32_hal_uart_error_isr(navmin_stm32_hal_t *hardware)
{
    if (hardware != NULL) {
        hardware->rx_armed = false;
        (void)navmin_stm32_hal_start_rx(hardware);
    }
}

bool navmin_stm32_hal_pop_rx(
    navmin_stm32_hal_t *hardware,
    uint8_t *byte,
    uint32_t *timestamp_ms
)
{
    uint16_t tail;

    if (hardware == NULL || byte == NULL || timestamp_ms == NULL) {
        return false;
    }

    tail = hardware->rx_tail;
    if (tail == hardware->rx_head) {
        return false;
    }

    __DMB();
    *byte = hardware->rx_bytes[tail];
    *timestamp_ms = hardware->rx_timestamps_ms[tail];
    hardware->rx_tail = (uint16_t)(
        (tail + UINT16_C(1)) & NAVMIN_STM32_RX_RING_MASK);
    return true;
}
