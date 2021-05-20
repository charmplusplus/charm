-include ../../../../common.mk

OPTS = -O3
CHARMC = ../../../../../bin/ampicxx $(OPTS)
MPIOPTS = -O3 -std=c++11
MPICC = mpicxx $(MPIOPTS)

TARGET = jacobi3d
OBJS = $(TARGET).o $(TARGET)CUDA.o
MPI_OBJS = $(TARGET)_mpi.o $(TARGET)CUDA.o

all: $(OBJS)
	$(CHARMC) -o $(TARGET) $^

mpi: $(MPI_OBJS)
	$(MPICC) -o $(TARGET)_mpi $^ -lcuda -lcudart -L$(CUDA_DIR)/lib64

$(TARGET).o: $(TARGET).C
	$(CHARMC) -o $@ -c $<

$(TARGET)_mpi.o: $(TARGET).C
	$(MPICC) -o $@ -c $<

$(TARGET)CUDA.o: $(TARGET).cu $(TARGET).h
	nvcc -o $@ -c $<

clean:
	rm -rf charmrun conv-host moduleinit* *.o $(TARGET) $(TARGET)_mpi *~ *.sts core ampirun
