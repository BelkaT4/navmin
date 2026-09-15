# NavMin STM32 firmware

This tree is the single STM32 firmware tree for BelkaT4 / NavMin. It keeps the hardware-independent protocol/control code in the STM32CubeIDE-compatible `Core/Inc` + `Core/Src` layout so later target HAL glue can reuse the same implementation without a protocol or control fork.

## Implemented checkpoints

### Stage 4A — protocol foundation

`navmin_protocol.c` implements the authoritative v1 wire foundation from `docs/dev/architecture/serial-protocol.md`:

- `AA 55` request framing and stream resynchronization;
- `LENGTH` validation and maximum 255-byte frames;
- CRC-16/MODBUS;
- 20 ms inter-byte timeout for incomplete candidates;
- response encoding with an empty v1 `EVENTS` section;
- ordinary `expected_request_id` sequencing with uint16 wrap;
- one exact retry cache for the last transaction;
- `EMERGENCY_STOP` safety/resynchronization ordering.

`PING` is handled directly by the protocol core. Recognized non-service commands are delegated only after protocol sequence/payload framing checks. Emergency uses its dedicated callback because the wire contract requires the safety side effect before malformed-payload validation.

### Stage 4B1 — control state / config / motors / Emergency

`navmin_control.c` is hardware-independent and currently owns only:

- reset state: `configured=false`, motors OFF, logical motion state cleared;
- full 20-byte little-endian `SET_CONFIG` decode/validation;
- one aggregate config snapshot and `configured` flag;
- `MOTOR_ON` / `MOTOR_OFF` semantics;
- the common immediate hard-stop primitive used by `MOTOR_OFF` and Emergency;
- a logical driver-enable callback boundary;
- executor/emergency callbacks compatible with the accepted Stage 4A protocol core.

`SET_VELOCITY` is implemented by Stage 4B2b2 below and `MOVE_RELATIVE` by Stage 4B3. `SET_BAUDRATE` intentionally still does not receive fake success from the motion/control owner; Stage 4C implements it in the hardware-specific application coordinator that owns USART reconfiguration.

`MOTOR_OFF` and Emergency share one control hard-stop boundary. Since Stage 4B2b1 this boundary delegates numeric velocity/target/phase clearing to the single owned `navmin_motion_t`, then clears only control-level intent flags. `MOTOR_OFF` additionally disables drivers; Emergency deliberately preserves driver and motor ON/OFF state and creates no persistent latch. Repeated `MOTOR_ON`/`MOTOR_OFF` remain idempotent at the driver callback boundary.


### Stage 4B2a — velocity motion primitive / acceleration / STEP scheduler

`navmin_motion.c` adds the hardware-independent numeric motion primitive used by Stage 4B2b2. Its acceleration limiter and STEP scheduler remain protocol-agnostic; `navmin_control` owns wire command acceptance, watchdog timing and the control-tick integration.

The control update rate is fixed at compile time:

```text
NAVMIN_CONTROL_RATE_HZ = 20,000 Hz
```

Velocity uses signed fixed-point `int32_t` units of `1 / 20,000 steps/s`:

```text
velocity_q = velocity_steps_s * 20,000
```

This representation removes a high-rate acceleration division/remainder. For one 20 kHz tick:

```text
delta_velocity_q = acceleration_steps_s2
```

so every integer configured acceleration is represented exactly. Reversal braking clamps at exact zero and deliberately discards any unused acceleration budget for that tick; opposite-sign acceleration starts only on a later control tick. No floating-point arithmetic is used in the high-rate path.

STEP scheduling uses a signed `int32_t` phase accumulator. Each tick adds `current_velocity_q`; one STEP is requested when phase reaches:

```text
NAVMIN_STEP_PHASE_THRESHOLD = 20,000 * 20,000 = 400,000,000
```

At the firmware speed ceiling of 10,000 steps/s, one tick contributes at most 200,000,000 phase units, so an axis can cross at most one STEP threshold per tick. X and Y are scheduled independently and may each emit one STEP during the same tick. One callback invocation represents exactly one STEP request with axis and signed direction. Commanded position is signed `int64_t` and changes only when that callback is actually invoked.

At the supported numeric maxima:

```text
max |velocity_q|       = 10,000 * 20,000 = 200,000,000
max acceleration delta = 100,000 velocity_q units/tick
phase threshold         = 400,000,000
max phase before fold   < 600,000,000
```

All hot-path velocity/acceleration/phase values therefore remain well inside signed 32-bit range (`2,147,483,647`). `int64_t` is used only for commanded position accounting. At 10,000 emitted steps/s, exhausting signed 64-bit position would require roughly 29 million years of continuous one-direction operation.

Dynamic motion limits are supplied as an already-validated snapshot. Lowering max speed clips only the currently stored effective target; current velocity decelerates through the limiter. Raising max speed does not resurrect an older clipped request. Changing acceleration needs no fractional-history reset because this representation has no acceleration remainder; the new integer acceleration increment is used on the next control tick.

When current velocity reaches stable zero, the per-axis STEP phase is cleared. `navmin_motion_hard_stop()` immediately clears current velocity, target velocity and STEP phase on both axes while preserving commanded position. This prevents latent fractional phase from creating a STEP after stop/restart.

### Stage 4B2b1 — control ↔ motion wiring

`navmin_control_t` now owns exactly one `navmin_motion_t`. Control owns configured/motor/config/behaviour state; motion remains the sole owner of numeric current/target velocity, STEP phase, commanded position and numeric motion limits. No duplicate numeric motion state is added to control.

`navmin_control_hardware_t` uses one hardware-independent context for three boundaries:

- logical driver enable/disable;
- individual STEP requests emitted by `navmin_motion`;
- optional `enter_critical` / `exit_critical` callbacks.

There are still no HAL, GPIO, TIM, UART or interrupt-register calls in this layer. Stage 4C provides those target callback implementations under `Target/` without changing the hardware-independent control API.

A valid `SET_CONFIG` is decoded and fully validated before publication. Its motion limits are derived from the same local candidate. One critical-section boundary then performs:

```text
enter critical
→ navmin_motion_set_limits(full next motion limits)
→ control.config = full next config snapshot
→ configured = true
→ exit critical
```

This prevents a future timer/ISR reader from observing a new control snapshot with old motion limits or the reverse. If `navmin_motion_set_limits()` unexpectedly rejects the already validated candidate, it returns before mutating motion state; control exits the boundary with the previous config/motion snapshot intact and reports `INTERNAL_ERROR`.

Dynamic speed/acceleration semantics therefore come directly from the accepted motion primitive: lowering max speed clips only the stored effective target while current velocity remains continuous; raising max speed does not resurrect a previously clipped request; a new acceleration limit is used on the next motion update. Stage 4B2b2 adds the watchdog semantics for the already atomic `velocity_watchdog_timeout_ms` field without changing the config publication boundary.

`navmin_control_init()` initializes the owned motion state with no valid limits and passes the common hardware STEP sink into it. `MOTOR_OFF` and Emergency call `navmin_motion_hard_stop()`, preserving commanded position while clearing current velocity, target velocity and STEP phase immediately. `MOTOR_ON` never restores old numeric motion.

### Stage 4B2b2 — SET_VELOCITY / watchdog / control tick

The Stage 4A executor callback now receives the `uint32_t now_ms` value from the protocol parser when a complete CRC-valid ordinary request is actually processed. Exact retries are still returned from the retry cache before the executor, so a transport retry never refreshes the velocity watchdog.

`SET_VELOCITY` decodes two little-endian signed `int32_t` values. Every `int32_t` input is accepted as a wire value and the existing motion primitive clamps the effective target to the current configured max speed. A successful command atomically publishes the clamped target, velocity-control intent, clears any relative intent, stores `last_velocity_setpoint_ms`, and arms the watchdog under the existing hardware-independent critical-section callbacks. `SET_VELOCITY(0,0)` is an ordinary setpoint: it leaves motors/drivers enabled and decelerates through the acceleration limiter.

The control-owned watchdog state is only:

```text
velocity_watchdog_armed
last_velocity_setpoint_ms
```

`navmin_control_tick(control, now_ms)` represents exactly one 20 kHz control update. It first evaluates wrap-safe unsigned elapsed time:

```text
elapsed = now_ms - last_velocity_setpoint_ms
```

If the armed watchdog has expired (`elapsed >= velocity_watchdog_timeout_ms`), only the motion target is set to `(0,0)` and the watchdog is disarmed. The same call then performs exactly one `navmin_motion_control_tick()`, so deceleration remains acceleration-limited and STEP output may continue while the commanded velocity ramps down. Watchdog expiry is not a hard stop, does not disable drivers and does not clear the velocity-control intent.

Dynamic `SET_CONFIG` does not alter the saved setpoint timestamp or watchdog armed state. A timeout decrease can therefore make the existing watchdog age expire on the next control tick; a timeout increase preserves the same age. Max-speed and acceleration changes keep the previously accepted Stage 4B2b1 semantics. `MOTOR_OFF` and Emergency disarm the watchdog inside their existing atomic hard-stop boundary, while `MOTOR_ON` does not re-arm it.

### Stage 4B3 — MOVE_RELATIVE / relative planner

`MOVE_RELATIVE` decodes two little-endian signed `int32_t` deltas and applies the fixed firmware sanity bounds:

```text
NAVMIN_MAX_RELATIVE_DELTA_X_STEPS = 100,000
NAVMIN_MAX_RELATIVE_DELTA_Y_STEPS = 100,000
```

The bounds protect the firmware from anomalous payloads; they are not mechanical absolute limits and do not replace the PC-side `max-relative-move-deg` check. Magnitude checks promote to `int64_t`, so `INT32_MIN` is rejected without signed overflow.

A relative target is stored as signed `int64_t` commanded-position coordinates. Acceptance computes each target from the actual commanded position at that moment:

```text
relative_target_x = motion.x.commanded_position_steps + delta_x
relative_target_y = motion.y.commanded_position_steps + delta_y
```

The additions are checked before publication; an unrepresentable `int64_t` target returns `INTERNAL_ERROR` without partial state mutation. Successful acceptance is one existing critical-section publication: the new X/Y targets become active, velocity-control intent is cleared, and the velocity watchdog is disarmed. Current velocity, STEP phase and commanded position are deliberately preserved, so replacement while moving is continuous rather than a hidden hard stop.

The relative planner runs independently per axis from `navmin_control_tick()`. It never emits STEP directly and never writes commanded position. Instead it chooses only a desired velocity target for the accepted `navmin_motion` primitive:

```text
direction to relative target
+ current velocity / STEP phase
+ current max-speed and acceleration limits
→ desired velocity {-max_speed, 0, +max_speed}
→ navmin_motion acceleration limiter
→ navmin_motion STEP scheduler
```

Braking uses bounded integer look-ahead. For the current velocity direction, the planner computes the exact future fixed-point phase accumulated if the motion target is changed to zero immediately. The sum is an arithmetic progression of future decelerating velocities, so no distance-dependent loop or floating point is needed. Dividing current directed phase plus that future phase by the STEP phase threshold gives the number of whole STEP requests still unavoidable under immediate braking. If that count is at least the remaining target distance, the planner requests velocity zero; otherwise it requests the current configured max speed toward the target. The real velocity transition remains exclusively in `navmin_motion`.

With the accepted Stage 4B2a bounds, the largest velocity magnitude is 200,000,000 velocity-q units. Even with the minimum valid acceleration of 1 step/s², the arithmetic-progression intermediate is below about `2e16`, safely inside `uint64_t`/`int64_t`. Position-distance math uses unsigned difference so opposite-sign `int64_t` positions do not cause signed subtraction overflow.

A replacement `MOVE_RELATIVE` always rebases on the current commanded position, not the previous target. If the new target is behind current motion, the existing acceleration limiter brakes through exact zero before opposite acceleration; unavoidable transient overshoot is permitted and the planner then converges back to the target. `MOVE_RELATIVE(0,0)` is still a new relative target at the current position and does not hard-stop existing motion.

Relative motion has no communication watchdog. A successful `MOVE_RELATIVE` disarms the velocity watchdog; subsequent control ticks ignore stale velocity timeout state while the relative target owns motion. A later successful `SET_VELOCITY` clears the relative target and re-arms the velocity watchdog. Dynamic `SET_CONFIG` does not snapshot a profile: each relative control update uses the current motion max-speed and acceleration limits, so speed/acceleration changes apply to the active move without changing its position target.

Natural completion is internal only. The relative intent clears after both axes simultaneously satisfy:

```text
commanded_position_steps == relative target
current_velocity_q == 0
target_velocity_q == 0
step_phase_q == 0
```

No response/event is emitted at that point. The only wire response was the original `MOVE_RELATIVE → OK` acceptance response. Exact protocol retry is still intercepted before the executor, so retrying the same raw transaction never rebases the target from a later commanded position.

### Stage 4C — STM32F103 HAL target / UART / runtime baud

`Target/` adds the first real STM32F103C8T6 target around the unchanged hardware-independent protocol/control/motion core. The target is intentionally UART-first: the v1 binary protocol remains independent of a later RS485 physical migration.

Target baseline:

```text
MCU: STM32F103C8T6, LQFP48
clock: 8 MHz HSE -> PLL x9 -> 72 MHz SYSCLK
USART3: PB10 TX, PB11 RX, 8N1
startup baud: 9600
TIM2: 20 kHz control tick (PSC=71, ARR=49 at 72 MHz timer clock)
X: STEP PB7, DIR PB8, ENABLE PB9
Y: STEP PB4, DIR PB5, ENABLE PB6
ENABLE: active-low
X DIR low/high: right/left
Y DIR low/high: up/down
```

The GPIO adapter maps internal positive X/Y STEP direction to board-level DIR low (right/up). PC-side `invert` remains the owner of installation-specific sign inversion before degrees/velocity are converted into signed steps. There is no second position counter in the HAL adapter: commanded position is still changed only by `navmin_motion` after one `emit_step` callback.

The target starts with STEP low and ENABLE high before PB4..PB9 are configured as outputs, so motor drivers are disabled at reset/startup. PB12..PB15 are configured as pull-up inputs because they physically exist on the board, but v1 firmware deliberately implements no limit-switch policy.

#### UART RX/TX

USART3 RX uses one-byte HAL interrupt reception. The ISR does only bounded transport work: it timestamps each received byte with `HAL_GetTick()`, pushes byte+timestamp into a 256-slot single-producer/single-consumer ring, and immediately rearms RX. The main loop drains that ring into `navmin_protocol_feed_byte()`. Preserving the per-byte ISR timestamp means the Stage 4A 20 ms inter-byte timeout is based on actual byte arrival times rather than on when the main loop happens to drain the queue. A 256-slot ring has 255 usable entries, exactly enough for the maximum 255-byte protocol frame. Overflow drops the newest byte and increments a diagnostic counter; the parser then relies on its existing framing/CRC/resynchronization rules.

Responses use blocking `HAL_UART_Transmit()` only from main/command context, never from TIM2. The transport callback reports success only after the STM32F1 HAL has completed transmission and the USART `TC` flag is set. No debug text is written to USART3, so the physical stream remains binary protocol only. Motion timing continues in TIM2 interrupts while a response is transmitted.

#### TIM2 and STEP pulses

TIM2 is configured for one update interrupt every 50 us. `HAL_TIM_PeriodElapsedCallback()` performs exactly one `navmin_control_tick()` for each TIM2 update. Motion scheduling never runs from the main loop.

`emit_step()` sets the board DIR level, waits a fixed 1 us DIR setup interval, drives STEP high for 2 us, then drives STEP low. These two constants are a conservative board-level assumption because the exact external STEP/DIR driver has not yet been fixed in the architecture. They are intentionally named in `navmin_stm32_hal.h` and must be checked against the actual driver datasheet and an oscilloscope during Stage 8 physical verification. No millisecond delay is used in the 20 kHz ISR. At most one pulse per axis can be requested per tick by the accepted 10,000 steps/s firmware bound; if both axes pulse in one tick, the current implementation spends about 6 us in explicit DIR/pulse waits plus GPIO/planner overhead.

#### Critical sections

The existing hardware-independent `enter_critical` / `exit_critical` callbacks are mapped to PRIMASK on STM32. Entry saves the previous PRIMASK at the outermost level and disables IRQs; a small depth counter makes the adapter safe if these short publication boundaries ever nest. Exit restores IRQ enable only when the matching outermost entry observed interrupts enabled. The protected control/config publications are short and contain no UART transmission or other long HAL operation.

#### `SET_BAUDRATE`

The supported v1 whitelist is exactly:

```text
9600, 19200, 38400, 57600, 115200
```

The hardware-specific application coordinator owns runtime baud switching; `navmin_control` remains unaware of UART. `SET_BAUDRATE` is accepted only while motors are OFF. Repeating the current baud returns `OK` without reconfiguration. For an actual switch the sequence is strictly:

```text
request arrives at old baud
-> protocol/executor accepts and records pending new baud
-> complete response OK is transmitted at old baud
-> physical USART TX complete / TC observed
-> RX is stopped and ring is cleared
-> USART3 is reinitialized at new baud
-> RX interrupt is rearmed
```

If physical response transmission fails, the switch remains pending and the UART stays at the old baud. An exact protocol retry is served from the Stage 4A cache without re-executing the command; after that cached response is physically transmitted at the old baud, the pending switch is applied. After a successful switch an exact retry may therefore receive the cached response at the STM32's current/new baud, matching the authoritative recovery contract. A new accepted ordinary transaction (including protocol-core-owned `PING`/parse-error paths) or a new Emergency invalidates an older unconfirmed pending switch, because the one-entry exact-retry cache has moved on and that old baud transaction can no longer be completed. `INVALID_REQUEST_ID` alone does not cancel it, so the valid exact retry remains possible. If reinitializing the requested baud fails, the adapter attempts to restore the previous UART baud and RX reception.

#### Target build

The project does not vendor the full STM32CubeF1 HAL tree. Supply an official STM32CubeF1 package root through `STM32CUBE_F1_DIR`. The user-supplied legacy `.ioc` identifies its firmware package as **STM32Cube FW_F1 V1.8.7** (CubeMX 6.5.0); that is the reproducibility reference for this target and the version to use for the Stage 4C build. The target code uses only the corresponding STM32F1 HAL/CMSIS API surface. `Target/Startup/startup_stm32f103c8tx.s`, `Target/STM32F103C8TX_FLASH.ld` (64 KiB Flash / 20 KiB RAM), and `Target/Core/Src/system_stm32f1xx.c` are the minimal generated/CMSIS target-reference pieces retained from the user-supplied legacy STM32F103C8 project. No legacy application, serial parser, command state machine, completion protocol, or limit-switch behavior is copied.

Required build tools are GNU Arm Embedded binaries named `arm-none-eabi-gcc`, `arm-none-eabi-objcopy`, `arm-none-eabi-objdump`, and `arm-none-eabi-size`. The target build always uses Cortex-M3 Thumb soft-float flags and `-O2`; optimization is not optional because the 20 kHz control path is part of the target timing budget.

From the project root:

```bash
make -C firmware/stm32 target-build \
  STM32CUBE_F1_DIR=/absolute/path/to/STM32CubeF1
```

Equivalent direct command:

```bash
make -C firmware/stm32/Target \
  STM32CUBE_F1_DIR=/absolute/path/to/STM32CubeF1
```

Expected outputs:

```text
firmware/stm32/build/target/navmin_stm32f103c8.elf
firmware/stm32/build/target/navmin_stm32f103c8.bin
firmware/stm32/build/target/navmin_stm32f103c8.hex
firmware/stm32/build/target/navmin_stm32f103c8.map
```

`make -C firmware/stm32/Target disasm ...` additionally writes the annotated target listing.

The Stage 4B3 planner intentionally retains its accepted bounded 64-bit integer look-ahead. Optimized Cortex-M3 compiler output must be inspected because software 64-bit division helpers can appear on this MCU. A real worst-case ISR duration cannot be claimed without the target toolchain and/or hardware cycle measurement; scope measurement of TIM2 ISR execution against the 50 us budget remains a required Stage 8 physical verification. The planner is not redesigned merely from static suspicion without a measured/proven overrun.

RS485, DE/RE, turnaround timing, termination/biasing, half-duplex arbitration and a second protocol version are not implemented. A future RS485 migration must replace only this low-level byte transport.

## Stage 4B1 firmware sanity bounds

These are compile-time implementation-protection limits, not mechanical absolute limits and not `config.json` defaults:

```text
NAVMIN_MAX_SUPPORTED_STEP_RATE             = 10,000 steps/s
NAVMIN_MAX_SUPPORTED_ACCELERATION          = 100,000 steps/s²
NAVMIN_MAX_SUPPORTED_WATCHDOG_TIMEOUT_MS   = 60,000 ms
```

Rationale:

- The legacy project is evidence that a 20 kHz control reference and about 8,000 steps/s were previously practical on STM32F103C8T6. A 10,000 steps/s ceiling stays at half of that 20 kHz reference, leaving scheduling/pulse-width margin for the implemented 20 kHz scheduler/target glue instead of claiming the theoretical one-step-per-tick limit.
- The legacy acceleration ceiling was 50,000 steps/s². `100,000 steps/s²` remains small relative to 32-bit arithmetic. With the Stage 4B2a 20 kHz fixed-point representation it contributes exactly 100,000 velocity-q units per tick, i.e. 5 steps/s of commanded-velocity change per tick.
- `60,000 ms` is far above the expected operational watchdog range (hundreds of milliseconds) while providing a clear bounded sanity ceiling and remaining trivial for future wrap-safe `uint32_t` elapsed-time arithmetic.

Stage 4B3 adds separate static relative-delta sanity bounds of ±100,000 steps per axis. They are implementation protection only and remain distinct from PC-side mechanical `max-relative-move-deg`.

## Current layout

```text
firmware/stm32/
├── Core/
│   ├── Inc/
│   │   ├── navmin_control.h
│   │   ├── navmin_motion.h
│   │   └── navmin_protocol.h
│   └── Src/
│       ├── navmin_control.c
│       ├── navmin_motion.c
│       └── navmin_protocol.c
├── Target/
│   ├── Core/Inc/            # minimal HAL/Cube-compatible target headers
│   ├── Core/Src/            # clock/GPIO/TIM/UART init + IRQ glue
│   ├── Inc/                 # STM32 app/transport adapters
│   ├── Src/                 # SET_BAUDRATE coordinator + HAL adapter
│   ├── Startup/             # STM32F103 startup assembly
│   ├── STM32F103C8TX_FLASH.ld
│   └── Makefile             # GNU Arm Embedded target build
├── Tests/host/
│   ├── test_navmin_control.c
│   ├── test_navmin_motion.c
│   ├── test_navmin_protocol.c
│   └── test_navmin_stm32_app.c
├── Makefile
└── README.md
```

`Core/` stays HAL-independent. Only `Target/` includes STM32 HAL/CMSIS headers. The host tests continue to compile with a native C compiler and do not require STM32 hardware or vendor headers.

## Host tests

A native C compiler is sufficient for the host suite; STM32CubeIDE, vendor HAL sources and an ARM cross-toolchain are not required for these tests.

From the project root:

```bash
make -C firmware/stm32 host-test
```

This builds and executes the Stage 4A protocol tests, the cumulative Stage 4B control/integration tests (including velocity/watchdog and relative planner behavior), the standalone Stage 4B2a motion primitive tests, and the hardware-independent Stage 4C application/`SET_BAUDRATE` coordinator tests.

To remove generated host binaries:

```bash
make -C firmware/stm32 clean
```

Build output is generated under `firmware/stm32/build/` and is ignored by the project-wide `build/` rule.

## Physical verification and deferred transport work

Software/host verification can cover protocol/control/motion behavior, target source compilation and reproducible target build mechanics. The following checks require the real STM32/driver/serial environment and therefore remain Stage 8 physical verification rather than ordinary unit tests:

- flash/boot STM32F103C8T6 and confirm 72 MHz clock setup;
- USART3 9600 8N1 binary `PING`/startup smoke on PB10/PB11;
- X/Y ENABLE active-low behavior and safe disabled startup;
- X/Y DIR polarity (`right/up` at DIR low, `left/down` at DIR high);
- STEP 1 us DIR setup / 2 us high pulse width and 20 kHz scheduling on an oscilloscope;
- worst-case TIM2 ISR duration, especially active two-axis relative planning, against the 50 us budget;
- real motor movement and commanded direction;
- runtime baud transitions across every supported rate, including lost-response recovery;
- Emergency latency and preservation of driver ON/OFF semantics;
- reconnect and host-side UART integration against the physical board.

The PB12..PB15 limit inputs have no v1 control policy. Encoders, homing, absolute positioning, completion events, `MOVE_COMPLETED`, `command_id`, persistent Emergency latch and generic firmware queues remain unimplemented by design.

RS485 remains a separate post-v1 physical migration. This target contains no RS485 transceiver control, DE/RE, half-duplex turnaround state, arbitration, termination/bias logic or RS485-specific wire semantics.
