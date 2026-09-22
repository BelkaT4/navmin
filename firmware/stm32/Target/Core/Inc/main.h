#ifndef NAVMIN_STM32_MAIN_H
#define NAVMIN_STM32_MAIN_H

#include "stm32f1xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

extern TIM_HandleTypeDef htim2;
extern UART_HandleTypeDef huart3;

void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif
