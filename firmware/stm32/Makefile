CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -ICore/Inc
HOST_BUILD_DIR := build/host
HOST_PROTOCOL_TEST := $(HOST_BUILD_DIR)/test_navmin_protocol
HOST_CONTROL_TEST := $(HOST_BUILD_DIR)/test_navmin_control
PROTOCOL_SOURCE := Core/Src/navmin_protocol.c
CONTROL_SOURCE := Core/Src/navmin_control.c

.PHONY: host-test clean

host-test: $(HOST_PROTOCOL_TEST) $(HOST_CONTROL_TEST)
	$(HOST_PROTOCOL_TEST)
	$(HOST_CONTROL_TEST)

$(HOST_PROTOCOL_TEST): $(PROTOCOL_SOURCE) Tests/host/test_navmin_protocol.c Core/Inc/navmin_protocol.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PROTOCOL_SOURCE) Tests/host/test_navmin_protocol.c -o $(HOST_PROTOCOL_TEST)

$(HOST_CONTROL_TEST): $(PROTOCOL_SOURCE) $(CONTROL_SOURCE) Tests/host/test_navmin_control.c Core/Inc/navmin_protocol.h Core/Inc/navmin_control.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PROTOCOL_SOURCE) $(CONTROL_SOURCE) Tests/host/test_navmin_control.c -o $(HOST_CONTROL_TEST)

clean:
	rm -rf build
