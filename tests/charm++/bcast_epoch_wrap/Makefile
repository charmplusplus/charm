-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: bcast_epoch_wrap

bcast_epoch_wrap.decl.h bcast_epoch_wrap.def.h: bcast_epoch_wrap.ci
	$(CHARMC) bcast_epoch_wrap.ci

bcast_epoch_wrap: bcast_epoch_wrap.C bcast_epoch_wrap.decl.h bcast_epoch_wrap.def.h
	$(CHARMC) -language charm++ bcast_epoch_wrap.C -o bcast_epoch_wrap

# Epochs that wrap at 2^16 (three wraps), then the full range. The 9-bit run
# has bursts longer than the interval between cleaning passes (64 broadcasts)
# and no migration, which such a small range does not allow.
test: bcast_epoch_wrap
	$(call run, ./bcast_epoch_wrap +p4 +bcastEpochBits 16)
	$(call run, ./bcast_epoch_wrap +p1 +bcastEpochBits 16)
	$(call run, ./bcast_epoch_wrap +p4 20000)
	$(call run, ./bcast_epoch_wrap +p4 +bcastEpochBits 9 20000 100 0)

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o bcast_epoch_wrap charmrun
