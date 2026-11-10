-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: abort

abort: abort.o
	$(CHARMC) -o abort abort.o

abort.o: abort.C
	$(CHARMC) -std=c++20 -c abort.C

clean:
	rm -f *.o abort charmrun conv-host ampirun

# MPI_Abort's error code becomes the exit status when rank 3 (in the second
# process under TESTRUN_PROCS=2) aborts while the others block in a receive,
# and when every rank aborts at once.
test: abort
	$(call run, ./abort +p2 +vp4 3 9 2>&1); rc=$$?; [ $$rc -eq 9 ] && echo "campi abort 3 9 test passed." || (echo "campi abort 3 9 test failed (exit code was $$rc, but expected 9)."; false)
	$(call run, ./abort +p2 +vp4 all 5 2>&1); rc=$$?; [ $$rc -eq 5 ] && echo "campi abort all 5 test passed." || (echo "campi abort all 5 test failed (exit code was $$rc, but expected 5)."; false)
