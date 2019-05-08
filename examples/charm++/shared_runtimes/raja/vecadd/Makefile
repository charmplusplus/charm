-include ../../../../common.mk
-include ../Makefile.common

TARGET = vecadd

all: $(TARGET)

$(TARGET): $(TARGET)_raja.o $(TARGET)_charm.o
	$(CHARMC) -language charm++ -fopenmp -L$(RAJA_LIB) -lRAJA -L$(CUDATOOLKIT_HOME)/lib64 -lcudart -o $@ $^

$(TARGET)_raja.o: $(TARGET)_raja.cpp $(TARGET).h
	nvcc $(NVCC_OPTS) -I$(RAJA_INC) -c $<

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET)_charm.o: $(TARGET)_charm.C $(TARGET).decl.h $(TARGET).h
	$(CHARMC) -c $< -I$(RAJA_INC) -I$(CUDATOOLKIT_HOME)/include

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun

test: all
	$(call run, ./$(TARGET) +p2)
