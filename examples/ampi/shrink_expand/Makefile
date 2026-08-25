-include ../../common.mk
CHARMC=../../../bin/ampicc $(OPTS)

all: jacobi1d subcomm

# No globals privatization: this program keeps its state on the heap and stack,
# both of which travel with the rank through Isomalloc. Add -pieglobals if the
# program grows globals that have to be per-rank.
jacobi1d: jacobi1d.c
	$(CHARMC) -module CommonLBs -o jacobi1d jacobi1d.c

# Several reduction trees per iteration, on communicators with deliberately
# different membership -- the case a single MPI_COMM_WORLD reduction does not
# reach.
subcomm: subcomm.c
	$(CHARMC) -module CommonLBs -o subcomm subcomm.c

clean:
	rm -f *.o jacobi1d subcomm *.user.so charmrun conv-host
