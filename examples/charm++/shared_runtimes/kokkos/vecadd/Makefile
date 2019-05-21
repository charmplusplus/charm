-include ../../../../common.mk
-include ../Makefile.common

TARGET = vecadd

all: $(TARGET)

$(TARGET): $(TARGET)_kokkos.o $(TARGET)_charm.o
	$(CHARMC) -language charm++ -fopenmp -L$(KOKKOS_CUDA_LIB) -lkokkos -L$(CUDATOOLKIT_HOME)/lib64 -lcudart -o $@ $^

$(TARGET)_kokkos.o: $(TARGET)_kokkos.cpp $(TARGET).h
	$(NVCC_WRAPPER) $(NVCC_OPTS) -c -I$(KOKKOS_CUDA_INC) $<

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET)_charm.o: $(TARGET)_charm.C $(TARGET).decl.h $(TARGET).h
	$(CHARMC) -c $< -I$(KOKKOS_CUDA_INC) -I$(CUDATOOLKIT_HOME)/include

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun

test: all
	$(call run, ./$(TARGET) +p2)
