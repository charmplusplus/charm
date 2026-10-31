-include ../../../../common.mk

OPTS = -O3

# Override to build against a different Charm++ build, e.g.
#   make CHARM_DIR=../../../../../reconverse-linux-x86_64-amd
CHARM_DIR ?= ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
LD_LIBS =

# GPU backend to build for: "cuda" or "hip". Defaults to whatever the Charm++
# build in CHARM_DIR was configured with -- cmake appends that backend's
# conv-mach header to conv-mach-opt.h -- so a recursive build in either tree
# needs no flags. Override with GPU=cuda or GPU=hip.
GPU ?= $(if $(shell grep -l conv-mach-hip.h $(CHARM_DIR)/include/conv-mach-opt.h 2>/dev/null),hip,cuda)

ifeq ($(GPU),hip)
  # charmc is a plain host compiler here, so user code needs the platform macro
  # spelled out; hipcc sets it for itself.
  GPU_FLAGS = -D__HIP_PLATFORM_AMD__=ON
else ifeq ($(GPU),cuda)
  GPU_FLAGS =
else
  $(error GPU must be one of "cuda" or "hip", got "$(GPU)")
endif

TARGET = latency
all: $(TARGET)

OBJS = $(TARGET).o

$(TARGET): $(OBJS)
	$(CHARMC) $(GPU_FLAGS) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET)-proj: $(OBJS)
	$(CHARMC) $(GPU_FLAGS) -language charm++ -tracemode projections -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $(GPU_FLAGS) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) $(GPU_FLAGS) -c $<

test: all
	$(call run, +p2 ./latency +gpumap block +gpushm)

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
