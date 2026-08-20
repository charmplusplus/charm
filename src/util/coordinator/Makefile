CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter

all: charm_coordinator coordinator_test_client

charm_coordinator: coordinator.cpp wire.h protocol.h
	$(CXX) $(CXXFLAGS) -o $@ coordinator.cpp

coordinator_test_client: test_client.cpp wire.h protocol.h coord_client.h
	$(CXX) $(CXXFLAGS) -o $@ test_client.cpp

clean:
	rm -f charm_coordinator coordinator_test_client

.PHONY: all clean
