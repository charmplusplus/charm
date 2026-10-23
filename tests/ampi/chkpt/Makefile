-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/ampicc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c $(OPTS)

# Runs only on macOS. The condition comes from a guard meant to skip
# multicore-darwin builds, which segfaulted on restart (issue #3555, closed),
# but as written it ran the test only on macOS. tests/ampi/Makefile leaves
# chkpt out on reconverse, so this target is not run there yet.
test: hello
  ifeq (1,$(CMK_MACOSX))
	    rm -rf log
	    $(call run, +p4 ./hello +vp8 )
	    -sync
	    $(call run, +p4 ./hello +vp8 +restart log )
	    $(call run, +p2 ./hello +vp8 +restart log )
	    -sync
	    -rm -fr log
	    $(call run, +p2 ./hello +vp8 )
	    -sync
	    $(call run, +p4 ./hello +vp8 +restart log )
  endif

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun ampirun
	rm -rf log
