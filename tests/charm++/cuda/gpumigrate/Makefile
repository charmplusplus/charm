# tests/charm++/cuda/gpumigrate -- device-state migration test.
#
# No .cu file: the buffers are filled and read back with plain device copies,
# because what is under test is whether device state survives a migration, not
# whether a kernel ran.
#
# CHARM_DIR defaults to ../../../.. , which is the Charm++ build root when this
# directory is reached through the build tree, the same convention the other
# tests use. Building straight out of the source tree needs it spelled out:
#
#   make CHARM_DIR=../../../../multicore-linux-x86_64-cuda

OPTS = -O2 -g

CHARM_DIR ?= ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)

TARGET = gpumigrate
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -o $@ $(TARGET).o $(LD_LIBS)

$(TARGET).decl.h $(TARGET).def.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) -c $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

# Two PEs in one process: the same-process transport only. IPC needs more than
# one process and RDMA more than one physical node, which a bare `make test`
# cannot arrange -- see README.txt.
test: all
	./$(TARGET) +pe 2
