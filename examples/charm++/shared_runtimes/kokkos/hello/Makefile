-include ../../../../common.mk
-include ../Makefile.common

TARGET = hello

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -fopenmp -L$(KOKKOS_OMP_LIB) -lkokkos -o $@ $^

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -fopenmp -I$(KOKKOS_OMP_INC) -c $<

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun

test: all
	$(call run, ./$(TARGET) +p2)
