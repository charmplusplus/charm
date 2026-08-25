-include ../../common.mk
CHARMC=../../../bin/ampicc $(OPTS)

all: jacobi1d

# No globals privatization: this program keeps its state on the heap and stack,
# both of which travel with the rank through Isomalloc. Add -pieglobals if the
# program grows globals that have to be per-rank.
jacobi1d: jacobi1d.c
	$(CHARMC) -module CommonLBs -o jacobi1d jacobi1d.c

clean:
	rm -f *.o jacobi1d *.user.so charmrun conv-host
