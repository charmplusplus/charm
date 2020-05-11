-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: longIdle

longIdle: longIdle.decl.h longIdle.def.h longIdle.C
	$(CHARMC) -language charm++ longIdle.C -o longIdle

longIdle.decl.h longIdle.def.h: longIdle.ci
	$(CHARMC) longIdle.ci

clean:
	rm -f *.decl.h *.def.h *.o longIdle charmrun

test: all
	$(call run, ./longIdle +p1 +longIdleThresh 0.1) # 0.1 is the threshold time for triggering LONG_IDLE
	$(call run, ./longIdle +p2 +longIdleThresh 0.1)

smptest:
	$(call run, ./longIdle +p2 ++ppn 2 +longIdleThresh 0.1)
	$(call run, ./longIdle +p4 ++ppn 2 +longIdleThresh 0.1)
