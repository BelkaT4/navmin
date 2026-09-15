#ifndef NAVMIN_CONTROL_H
#define NAVMIN_CONTROL_H

#include "navmin_motion.h"
#include "navmin_protocol.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stage 4B1 firmware sanity bounds. These are implementation protection
   limits, not mechanical absolute limits or user configuration defaults. */
#define NAVMIN_MAX_SUPPORTED_STEP_RATE UINT32_C(10000)
#define NAVMIN_MAX_SUPPORTED_ACCELERATION UINT32_C(100000)
#define NAVMIN_MAX_SUPPORTED_WATCHDOG_TIMEOUT_MS UINT32_C(60000)

typedef struct {
    uint32_t max_speed_x_steps_s;
    uint32_t max_speed_y_steps_s;
    uint32_t acceleration_x_steps_s2;
    uint32_t acceleration_y_steps_s2;
    uint32_t velocity_watchdog_timeout_ms;
} navmin_control_config_t;

typedef void (*navmin_set_drivers_enabled_fn)(void *context, bool enabled);
typedef void (*navmin_control_critical_fn)(void *context);

typedef struct {
    void *context;
    navmin_set_drivers_enabled_fn set_drivers_enabled;
    navmin_emit_step_fn emit_step;
    navmin_control_critical_fn enter_critical;
    navmin_control_critical_fn exit_critical;
} navmin_control_hardware_t;

typedef struct {
    navmin_control_config_t config;
    navmin_control_hardware_t hardware;
    navmin_motion_t motion;
    bool configured;
    bool motors_on;

    /* Control-level behaviour intents. Numeric velocity/phase/position state
       belongs only to motion and is cleared through the common hard stop. */
    bool velocity_target_present;
    bool relative_target_present;

    bool velocity_watchdog_armed;
    uint32_t last_velocity_setpoint_ms;
} navmin_control_t;

void navmin_control_init(
    navmin_control_t *control,
    navmin_control_hardware_t hardware
);

navmin_result_code_t navmin_control_execute_command(
    void *context,
    uint8_t command_code,
    const uint8_t *payload,
    uint8_t payload_length,
    uint32_t now_ms
);

void navmin_control_tick(navmin_control_t *control, uint32_t now_ms);

void navmin_control_emergency_stop(void *context);

navmin_protocol_executor_t navmin_control_protocol_executor(navmin_control_t *control);

#ifdef __cplusplus
}
#endif

#endif
