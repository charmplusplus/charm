OPTS = -O3 -DHAPI_CUDA_CALLBACK

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

NVCC = nvcc
NVCC_FLAGS = -c -std=c++11 -use_fast_math $(OPTS)
LD_LIBS = -module EveryLB -module DiffusionLB

TARGET = smload
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

# One block per GPU, the whole device: the ratio must come out 1.
test: all
	$(call run, +p4 ./$(TARGET) -W 2048 -H 2048 -w 1024 -h 1024 -i 30 -u 5 -f 10 -b 10 -s 84 -F +balancer DiffusionLB +LBDiffusionGpuDim)

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
