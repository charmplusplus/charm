-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/ampicc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c $(OPTS)

test: hello
  ifneq (1,$(CMK_MULTICORE))
    ifeq (1,$(CMK_MACOSX)) # Doesn't work on multicore-darwin (issue #3555)
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
  endif

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun ampirun
	rm -rf log
