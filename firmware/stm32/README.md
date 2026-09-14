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

The control module deliberately does **not** implement motion yet. `MOVE_RELATIVE`, `SET_VELOCITY`, and `SET_BAUDRATE` therefore never receive fake success from the 4B1 control owner; their real owners remain deferred to 4B2/4B3/4C.

`MOTOR_OFF` clears only minimal motion-intent presence flags before disabling drivers; 4B1 deliberately chooses no numeric velocity or position representation. Emergency performs the same hard motion clear but does not change driver enable state and creates no persistent latch. Repeated `MOTOR_ON`/`MOTOR_OFF` are idempotent at the driver callback boundary.

`SET_CONFIG` first decodes and validates a local candidate and then replaces `control->config` with one aggregate struct assignment. There is no concurrent high-rate control reader wired into `navmin_control` yet. Stage 4B2b will connect the accepted full snapshot to the motion primitive and add the eventual control-loop/ISR critical-section boundary around one complete replacement rather than mutate fields independently.


### Stage 4B2a — velocity motion primitive / acceleration / STEP scheduler

`navmin_motion.c` adds an isolated hardware-independent numeric motion primitive. It is intentionally not wired to the protocol `SET_VELOCITY` command yet; command acceptance, timestamps and the velocity watchdog remain Stage 4B2b.

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

## Stage 4B1 firmware sanity bounds

These are compile-time implementation-protection limits, not mechanical absolute limits and not `config.json` defaults:

```text
NAVMIN_MAX_SUPPORTED_STEP_RATE             = 10,000 steps/s
NAVMIN_MAX_SUPPORTED_ACCELERATION          = 100,000 steps/s²
NAVMIN_MAX_SUPPORTED_WATCHDOG_TIMEOUT_MS   = 60,000 ms
```

Rationale:

- The legacy project is evidence that a 20 kHz control reference and about 8,000 steps/s were previously practical on STM32F103C8T6. A 10,000 steps/s ceiling stays at half of that 20 kHz reference, leaving scheduling/pulse-width margin for the future 4B2/4C implementation instead of claiming the theoretical one-step-per-tick limit.
- The legacy acceleration ceiling was 50,000 steps/s². `100,000 steps/s²` remains small relative to 32-bit arithmetic. With the Stage 4B2a 20 kHz fixed-point representation it contributes exactly 100,000 velocity-q units per tick, i.e. 5 steps/s of commanded-velocity change per tick.
- `60,000 ms` is far above the expected operational watchdog range (hundreds of milliseconds) while providing a clear bounded sanity ceiling and remaining trivial for future wrap-safe `uint32_t` elapsed-time arithmetic.

Relative-delta firmware bounds are intentionally not defined in 4B1. They belong to 4B3 together with the actual relative planner/integration representation.

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
├── Tests/host/
│   ├── test_navmin_control.c
│   ├── test_navmin_motion.c
│   └── test_navmin_protocol.c
├── Makefile
└── README.md
```

The code has no dependency on STM32 HAL, UART registers, GPIO registers, timers, dynamic allocation, or RS485. Host tests replace driver enable with a fake callback.

## Host tests

A native C compiler is sufficient; STM32CubeIDE and an ARM cross-toolchain are not required for 4B1.

From the project root:

```bash
make -C firmware/stm32 host-test
```

This builds and executes the Stage 4A protocol tests, Stage 4B1 control tests, and Stage 4B2a motion tests.

To remove generated host binaries:

```bash
make -C firmware/stm32 clean
```

Build output is generated under `firmware/stm32/build/` and is ignored by the project-wide `build/` rule.

## Hardware baseline for later integration

The target integration remains fixed by the current Turret architecture and the legacy hardware reference:

```text
MCU: STM32F103C8T6, LQFP48
framework: STM32 HAL / CubeIDE-compatible
startup serial: 9600 8N1, full-duplex UART
USART3 TX: PB10
USART3 RX: PB11
X: STEP PB7, DIR PB8, ENABLE PB9
Y: STEP PB4, DIR PB5, ENABLE PB6
ENABLE: active-low
```

The driver-enable callback is logical (`true` = drivers enabled); active-low GPIO polarity is a Stage 4C HAL concern.

The legacy project used a 72 MHz HSE→PLL clock setup, TIM2 prescaler 71 / period 49, a historical 8,000 steps/s max-velocity bound, and 50,000 steps/s² max acceleration. These values are evidence only. No legacy serial parser, command set, application state machine, completion/status protocol, or limit-switch semantics are copied.

## Explicitly deferred

### Stage 4B2b

- wire `SET_VELOCITY` acceptance and control-owner wiring;
- protocol acceptance timestamp plumbing;
- velocity watchdog;
- dynamic `SET_CONFIG` integration with active velocity behaviour;
- exact-retry/watchdog integration.

### Stage 4B3

- relative target representation/integration;
- relative sanity bounds;
- relative planner, braking, target completion and reversal behaviour.

### Stage 4C

- STM32 HAL GPIO/TIM glue;
- USART3 RX/TX transport;
- physical `SET_BAUDRATE` switching;
- CubeIDE/ARM target build and reproducible target integration.

RS485/DE-RE, limit switches, encoders, absolute positioning, `MOVE_COMPLETED`, `RESET_ID`, `command_id`, asynchronous event FIFO, generic command FIFO, and persistent Emergency latch remain outside these implemented checkpoints.
