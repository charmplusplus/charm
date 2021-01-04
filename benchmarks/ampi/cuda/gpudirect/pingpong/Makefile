-include ../../../../common.mk

OPTS=-O3
MPIOPTS=-O3
CHARMC=../../../../../bin/ampicxx $(OPTS)
MPICC=mpicc $(MPIOPTS) # Should use 'cc' instead of 'mpicc' on Cray systems

all : pingpong-1way.c
	$(CHARMC) pingpong-1way.c -o pingpong
#	$(MPICC) pingpong-1way.c -o pingpong-mpi

clean:
	rm -rf charmrun conv-host moduleinit* *.o pingpong pingpong-mpi *~ *.sts core ampirun
