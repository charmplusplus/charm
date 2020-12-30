-include ../../../../common.mk

OPTS = -O3
CHARMC = ../../../../../bin/ampicxx $(OPTS)
MPIOPTS = -O3
MPICC = mpicc $(MPIOPTS)

TARGET = osu_latency

all : $(TARGET).c
	$(CHARMC) -o $(TARGET) $<
#	$(MPICC) -o $(TARGET)_mpi $<

clean:
	rm -rf charmrun conv-host moduleinit* *.o $(TARGET) $(TARGET)_mpi *~ *.sts core ampirun
