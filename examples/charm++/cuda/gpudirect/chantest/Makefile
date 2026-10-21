OPTS = -O2 -g

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

# The device compiler follows the build the runtime was made with (conv-mach-opt.mak):
# nvcc for a CUDA build, hipcc as HIP for a HIP build (kernels in CUDA syntax compile
# as HIP; the host side uses the hapi names of hapi_portable.h).
-include $(CHARM_DIR)/include/conv-mach-opt.mak
ifeq ($(BUILD_HIP),1)
NVCC = hipcc
NVCC_FLAGS = -x hip -O3 -c -std=c++17 -ffast-math $(patsubst %,--offload-arch=%,$(subst ;, ,$(HIP_ARCHS)))
else
NVCC = nvcc
NVCC_FLAGS = -O3 -c -std=c++17 -use_fast_math
endif
LD_LIBS =

TARGET = chantest
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
