OPTS = -O3 -g

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

NVCC = nvcc
NVCC_FLAGS = -O3 -c -std=c++11 -use_fast_math
LD_LIBS =

TARGET = persistent
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -balancer TreeLB -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
