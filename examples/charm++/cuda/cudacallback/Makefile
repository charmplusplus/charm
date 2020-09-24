-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

CHARM_INC = -I../../../../include
NVCC = nvcc
NVCC_OPTS = -std=c++11
LD_LIBS =

TARGET = cudacallback
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
