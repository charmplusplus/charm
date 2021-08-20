-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/ampicc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c $(OPTS)

test: hello
	rm -rf log
	$(call run, +p4 ./hello +vp8 )
ifeq (1,$(CMK_HAS_SYNC))
	-sync
endif
	$(call run, +p4 ./hello +vp8 +restart log )
	$(call run, +p2 ./hello +vp8 +restart log )
ifeq (1,$(CMK_HAS_SYNC))
	-sync
endif
	-rm -fr log
	$(call run, +p2 ./hello +vp8 )
ifeq (1,$(CMK_HAS_SYNC))
	-sync
endif
	$(call run, +p4 ./hello +vp8 +restart log )

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun ampirun
	rm -rf log
