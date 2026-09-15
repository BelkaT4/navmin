PROJECT := navmin_stm32f103c8
BUILD_DIR := ../build/target
CORE_DIR := ../Core
TARGET_CORE_DIR := Core

CROSS_COMPILE ?= arm-none-eabi-
CC := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump
SIZE := $(CROSS_COMPILE)size

ifeq ($(strip $(STM32CUBE_F1_DIR)),)
$(error STM32CUBE_F1_DIR is required; point it at an STM32CubeF1 package root)
endif

HAL_DIR := $(STM32CUBE_F1_DIR)/Drivers/STM32F1xx_HAL_Driver
CMSIS_DEVICE_DIR := $(STM32CUBE_F1_DIR)/Drivers/CMSIS/Device/ST/STM32F1xx
CMSIS_CORE_DIR := $(STM32CUBE_F1_DIR)/Drivers/CMSIS

CPPFLAGS := \
	-DUSE_HAL_DRIVER \
	-DSTM32F103xB \
	-I$(CORE_DIR)/Inc \
	-IInc \
	-I$(TARGET_CORE_DIR)/Inc \
	-I$(HAL_DIR)/Inc \
	-I$(HAL_DIR)/Inc/Legacy \
	-I$(CMSIS_DEVICE_DIR)/Include \
	-I$(CMSIS_CORE_DIR)/Include

CPUFLAGS := -mcpu=cortex-m3 -mthumb -mfloat-abi=soft
CFLAGS := $(CPUFLAGS) -std=gnu11 -O2 -g3 -ffunction-sections -fdata-sections \
	-fstack-usage -Wall -Wextra -Wshadow -Wdouble-promotion
ASFLAGS := $(CPUFLAGS) -x assembler-with-cpp
LDFLAGS := $(CPUFLAGS) -TSTM32F103C8TX_FLASH.ld \
	-Wl,--gc-sections -Wl,-Map=$(BUILD_DIR)/$(PROJECT).map \
	-specs=nano.specs -specs=nosys.specs
LDLIBS := -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

C_SOURCES := \
	$(CORE_DIR)/Src/navmin_protocol.c \
	$(CORE_DIR)/Src/navmin_control.c \
	$(CORE_DIR)/Src/navmin_motion.c \
	Src/navmin_stm32_app.c \
	Src/navmin_stm32_hal.c \
	$(TARGET_CORE_DIR)/Src/main.c \
	$(TARGET_CORE_DIR)/Src/stm32f1xx_hal_msp.c \
	$(TARGET_CORE_DIR)/Src/stm32f1xx_it.c \
	$(TARGET_CORE_DIR)/Src/system_stm32f1xx.c \
	$(HAL_DIR)/Src/stm32f1xx_hal.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_cortex.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_dma.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_flash.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_flash_ex.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_gpio.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_pwr.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_rcc.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_rcc_ex.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_tim.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_tim_ex.c \
	$(HAL_DIR)/Src/stm32f1xx_hal_uart.c

ASM_SOURCES := Startup/startup_stm32f103c8tx.s

OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o))) \
	$(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
DEPS := $(OBJECTS:.o=.d)

ELF := $(BUILD_DIR)/$(PROJECT).elf
BIN := $(BUILD_DIR)/$(PROJECT).bin
HEX := $(BUILD_DIR)/$(PROJECT).hex
LIST := $(BUILD_DIR)/$(PROJECT).list

vpath %.c $(sort $(dir $(C_SOURCES)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

.PHONY: all clean check-tools check-cube size disasm

all: check-tools check-cube $(ELF) $(BIN) $(HEX) size

check-tools:
	@command -v $(CC) >/dev/null || { echo "ERROR: $(CC) not found"; exit 1; }
	@command -v $(OBJCOPY) >/dev/null || { echo "ERROR: $(OBJCOPY) not found"; exit 1; }
	@command -v $(SIZE) >/dev/null || { echo "ERROR: $(SIZE) not found"; exit 1; }

check-cube:
	@test -f "$(HAL_DIR)/Src/stm32f1xx_hal.c" || { \
		echo "ERROR: STM32CubeF1 HAL not found under $(STM32CUBE_F1_DIR)"; exit 1; }
	@test -f "$(CMSIS_DEVICE_DIR)/Include/stm32f103xb.h" || { \
		echo "ERROR: STM32CubeF1 CMSIS device headers not found"; exit 1; }

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.s | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(ASFLAGS) -c $< -o $@

$(ELF): $(OBJECTS) STM32F103C8TX_FLASH.ld
	$(CC) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@

$(HEX): $(ELF)
	$(OBJCOPY) -O ihex $< $@

size: $(ELF)
	$(SIZE) $<

disasm: $(ELF)
	$(OBJDUMP) -d -S $< > $(LIST)
	@echo "Disassembly: $(LIST)"

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
