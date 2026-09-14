# NavMin STM32 firmware

This tree is the single STM32 firmware tree for BelkaT4 / NavMin.

Stage 4A contains only the hardware-independent protocol foundation. The source layout intentionally follows the STM32CubeIDE `Core/Inc` + `Core/Src` convention so the same C protocol core can be compiled into the later HAL target without copying or forking it.

## Stage 4A contents

```text
firmware/stm32/
├── Core/
│   ├── Inc/navmin_protocol.h
│   └── Src/navmin_protocol.c
├── Tests/host/test_navmin_protocol.c
├── Makefile
└── README.md
```

`navmin_protocol.c` implements the authoritative v1 wire foundation from `docs/dev/architecture/serial-protocol.md`:

- `AA 55` request framing and stream resynchronization;
- `LENGTH` validation and maximum 255-byte frames;
- CRC-16/MODBUS;
- 20 ms inter-byte timeout for incomplete candidates;
- response encoding with an empty v1 `EVENTS` section;
- ordinary `expected_request_id` sequencing with uint16 wrap;
- one exact retry cache for the last transaction;
- `EMERGENCY_STOP` safety/resynchronization ordering.

The protocol core has no dependency on STM32 HAL, UART registers, GPIO, timers, STEP generation, motor state, or RS485. Byte input is supplied by `navmin_protocol_feed_byte()`, time is supplied explicitly by the caller, and encoded response bytes leave through a callback. A later USART3 transport can therefore be only a byte-I/O adapter around this core.

`PING` is handled directly by the protocol foundation. Other recognized commands are payload-framed here and then delegated to the executor callback after protocol sequence checks. Their motor/config/velocity/baud semantics are intentionally deferred to Stage 4B/4C rather than given temporary Stage 4A behavior. `EMERGENCY_STOP` uses a dedicated safety callback because the protocol contract requires its side effect before payload validation.

## Host tests

A native C compiler is sufficient; STM32CubeIDE and an ARM cross-toolchain are not required for Stage 4A.

From the project root:

```bash
make -C firmware/stm32 host-test
```

To remove the generated host binary:

```bash
make -C firmware/stm32 clean
```

The host build output is generated under `firmware/stm32/build/` and is ignored by the project-wide `build/` rule.

## Hardware baseline for later integration

The later target integration is fixed by the current Turret architecture and confirmed by the legacy reference project:

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

The legacy project used a 72 MHz HSE→PLL clock setup and TIM2 with prescaler 71 / period 49. Those timer values are reference only, not Stage 4A protocol requirements or new control-loop bounds.

No legacy serial parser, command set, application state machine, binary build artifact, USART3 HAL transport, runtime baud switching, STEP generation, limit-switch behavior, RS485/DE/RE, or asynchronous event FIFO is copied into Stage 4A.

A reproducible STM32 target/cross-build is deliberately deferred to Stage 4C, where the actual USART3 HAL glue is introduced.
