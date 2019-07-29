-include ../../../common.mk
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: bcast_nonzero_root

OBJS = bcast_nonzero_root.o

bcast_nonzero_root: $(OBJS)
	$(CHARMC) -language charm++ -o bcast_nonzero_root $(OBJS)

cifiles: bcast_nonzero_root.ci
	$(CHARMC)  bcast_nonzero_root.ci
	touch cifiles

bcast_nonzero_root.o: bcast_nonzero_root.C cifiles
	$(CHARMC) -c bcast_nonzero_root.C

test: all
	$(call run, +p1 ./bcast_nonzero_root)
	$(call run, +p2 ./bcast_nonzero_root)
	$(call run, +p4 ./bcast_nonzero_root)
	$(call run, +p6 ./bcast_nonzero_root)
	$(call run, +p2 ./bcast_nonzero_root +noCMAForZC)
	$(call run, +p4 ./bcast_nonzero_root +noCMAForZC)
	$(call run, +p6 ./bcast_nonzero_root +noCMAForZC)

smptest: all
	$(call run, +p4 ./bcast_nonzero_root ++ppn 4)
	$(call run, +p6 ./bcast_nonzero_root ++ppn 3)
	$(call run, +p4 ./bcast_nonzero_root +noCMAforZC ++ppn 4)
	$(call run, +p6 ./bcast_nonzero_root +noCMAForZC ++ppn 3)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f bcast_nonzero_root charmrun cifiles
