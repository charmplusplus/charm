OPTS = -O3
CHARMC = ../../../bin/charmc -language ampi
MPICC = mpicc

all: mpibench mpibench.mpi

allgather: allgather.c
	$(CHARMC) -c allgather.c $(OPTS)
	$(CHARMC) -o allgather allgather.o $(OPTS) $(LIBS) -memory gnu

allgather.mpi: allgather.c
	$(MPICC) -c allgather.c $(OPTS)
	$(MPICC) -o allgather.mpi allgather.o $(OPTS) $(LIBS)

alltoall: alltoall.c
	$(CHARMC) -c alltoall.c $(OPTS)
	$(CHARMC) -o alltoall alltoall.o $(OPTS) $(LIBS) -memory gnu

alltoall.mpi: alltoall.c
	$(MPICC) -c alltoall.c $(OPTS)
	$(MPICC) -o alltoall.mpi alltoall.o $(OPTS) $(LIBS)

mpibench: mpibench.c
	$(CHARMC) -c mpibench.c $(OPTS)
	$(CHARMC) -o pgm mpibench.o $(OPTS) $(LIBS)

mpibench.mpi: mpibench.c
	$(MPICC) -c mpibench.c $(OPTS)
	$(MPICC) -o pgm-mpi mpibench.o $(OPTS) $(LIBS)

alltoall_VPtest: alltoall_VPtest.c
	$(CHARMC) -c alltoall_VPtest.c $(OPTS)
	$(CHARMC) -o alltoall_VPtest alltoall_VPtest.o $(OPTS) $(LIBS)

alltoall_VPtest_f: alltoall_VPtest.f
	$(CHARMC) -c alltoall_VPtest.f  -language ampif -swapglobals  $(OPTS)
	$(CHARMC) -o alltoall_VPtest_f alltoall_VPtest.o -language ampif -swapglobals $(OPTS) $(LIBS)

clean:
	rm -rf *~ *.o alltoall mpibench charmrun conv-host allgather
