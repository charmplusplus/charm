# tests/charm++/cuda/migmanifest -- the device migration manifest.
#
# CHARM_DIR defaults to ../../../.. , the Charm++ build root when this
# directory is reached through the build tree; from the source tree spell it
# out, as for gpumigrate:
#
#   make CHARM_DIR=../../../../multicore-linux-x86_64-cuda

OPTS = -O2 -g

CHARM_DIR ?= ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)

LD_LIBS = -module CommonLBs

TARGET = migmanifest
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -o $@ $(TARGET).o $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

# Two PEs in one process, with and without the device pool (+gpupool: the
# payload is received into an arena the buffers are rebound into; without it they are
# allocated and copied). The injection runs, which pass by aborting with the
# right message, are by hand; see migmanifest.C.
test: all
	./$(TARGET) +pe 2 +balancer RotateLB
	./$(TARGET) +pe 2 +balancer RotateLB +gpupool
