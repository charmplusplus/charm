-include ../../../../common.mk

OPTS = -O3
CHARMC = ../../../../../bin/ampicxx $(OPTS)
MPIOPTS = -O3
MPICC = mpicc $(MPIOPTS)

TARGET = jacobi3d

all: $(TARGET).C
	$(CHARMC) -o $(TARGET) $<

mpi: $(TARGET).C
	$(MPICC) -o $(TARGET)_mpi $< -lcuda -lcudart -L$(CUDA_DIR)/lib64

clean:
	rm -rf charmrun conv-host moduleinit* *.o $(TARGET) $(TARGET)_mpi *~ *.sts core ampirun
