OPTS = -O3

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

# The device compiler follows the build the runtime was made with (conv-mach-opt.mak):
# nvcc for a CUDA build, hipcc as HIP for a HIP build (kernels in CUDA syntax compile
# as HIP; the host side uses the hapi names of hapi_portable.h).
-include $(CHARM_DIR)/include/conv-mach-opt.mak
ifeq ($(BUILD_HIP),1)
NVCC = hipcc
NVCC_FLAGS = -x hip -c -std=c++17 -ffast-math $(OPTS) $(patsubst %,--offload-arch=%,$(subst ;, ,$(HIP_ARCHS)))
else
NVCC = nvcc
NVCC_FLAGS = -c -std=c++11 -use_fast_math $(OPTS)
endif
LD_LIBS =

TARGET = jacobi3d
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci $(TARGET).h
	$(CHARMC) $<

$(TARGET).def.h: $(TARGET).ci $(TARGET).h

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h $(TARGET).h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu $(TARGET).h
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
