CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -ICore/Inc
HOST_BUILD_DIR := build/host
HOST_TEST := $(HOST_BUILD_DIR)/test_navmin_protocol
HOST_SOURCES := Core/Src/navmin_protocol.c Tests/host/test_navmin_protocol.c

.PHONY: host-test clean

host-test: $(HOST_TEST)
	$(HOST_TEST)

$(HOST_TEST): $(HOST_SOURCES) Core/Inc/navmin_protocol.h
	mkdir -p $(HOST_BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(HOST_SOURCES) -o $(HOST_TEST)

clean:
	rm -rf build
