CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -ICore/Inc -ITarget/Inc
HOST_BUILD_DIR := build/host
HOST_PROTOCOL_TEST := $(HOST_BUILD_DIR)/test_navmin_protocol
HOST_CONTROL_TEST := $(HOST_BUILD_DIR)/test_navmin_control
HOST_MOTION_TEST := $(HOST_BUILD_DIR)/test_navmin_motion
HOST_STM32_APP_TEST := $(HOST_BUILD_DIR)/test_navmin_stm32_app
PROTOCOL_SOURCE := Core/Src/navmin_protocol.c
CONTROL_SOURCE := Core/Src/navmin_control.c
MOTION_SOURCE := Core/Src/navmin_motion.c
STM32_APP_SOURCE := Target/Src/navmin_stm32_app.c

.PHONY: host-test target-build target-clean clean

host-test: $(HOST_PROTOCOL_TEST) $(HOST_CONTROL_TEST) $(HOST_MOTION_TEST) $(HOST_STM32_APP_TEST)
	$(HOST_PROTOCOL_TEST)
	$(HOST_CONTROL_TEST)
	$(HOST_MOTION_TEST)
	$(HOST_STM32_APP_TEST)

$(HOST_PROTOCOL_TEST): $(PROTOCOL_SOURCE) Tests/host/test_navmin_protocol.c Core/Inc/navmin_protocol.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PROTOCOL_SOURCE) Tests/host/test_navmin_protocol.c -o $(HOST_PROTOCOL_TEST)

$(HOST_CONTROL_TEST): $(PROTOCOL_SOURCE) $(CONTROL_SOURCE) $(MOTION_SOURCE) Tests/host/test_navmin_control.c Core/Inc/navmin_protocol.h Core/Inc/navmin_control.h Core/Inc/navmin_motion.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PROTOCOL_SOURCE) $(CONTROL_SOURCE) $(MOTION_SOURCE) Tests/host/test_navmin_control.c -o $(HOST_CONTROL_TEST)

$(HOST_MOTION_TEST): $(MOTION_SOURCE) Tests/host/test_navmin_motion.c Core/Inc/navmin_motion.h Core/Inc/navmin_control.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MOTION_SOURCE) Tests/host/test_navmin_motion.c -o $(HOST_MOTION_TEST)

$(HOST_STM32_APP_TEST): $(PROTOCOL_SOURCE) $(CONTROL_SOURCE) $(MOTION_SOURCE) $(STM32_APP_SOURCE) Tests/host/test_navmin_stm32_app.c Core/Inc/navmin_protocol.h Core/Inc/navmin_control.h Core/Inc/navmin_motion.h Target/Inc/navmin_stm32_app.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PROTOCOL_SOURCE) $(CONTROL_SOURCE) $(MOTION_SOURCE) $(STM32_APP_SOURCE) Tests/host/test_navmin_stm32_app.c -o $(HOST_STM32_APP_TEST)

target-build:
	$(MAKE) -C Target all STM32CUBE_F1_DIR="$(STM32CUBE_F1_DIR)" $(if $(CROSS_COMPILE),CROSS_COMPILE="$(CROSS_COMPILE)",)

target-clean:
	$(MAKE) -C Target clean

clean:
	rm -rf build
