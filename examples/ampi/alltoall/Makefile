OPTS = -O3
CHARMC = ../../../bin/charmc -language ampi
MPICC = mpicc

all: alltoall alltoall_VPtest

alltoall: alltoall.c
	$(CHARMC) -c alltoall.c $(OPTS)
	$(CHARMC) -o alltoall alltoall.o $(OPTS) $(LIBS)

alltoall.mpi: alltoall.c
	$(MPICC) -c alltoall.c $(OPTS)
	$(MPICC) -o alltoall.mpi alltoall.o $(OPTS) $(LIBS)

mpibench: mpibench.c
	$(CHARMC) -c mpibench.c $(OPTS)
	$(CHARMC) -o mpibench mpibench.o $(OPTS) $(LIBS)

alltoall_VPtest: alltoall_VPtest.c
	$(CHARMC) -c alltoall_VPtest.c $(OPTS)
	$(CHARMC) -o alltoall_VPtest alltoall_VPtest.o $(OPTS) $(LIBS)

alltoall_VPtest_f: alltoall_VPtest.f
	$(CHARMC) -c alltoall_VPtest.f  -language ampif -swapglobals  $(OPTS)
	$(CHARMC) -o alltoall_VPtest_f alltoall_VPtest.o -language ampif -swapglobals $(OPTS) $(LIBS)

clean:
	rm -rf *~ *.o alltoall mpibench charmrun conv-host alltoall_VPtest
