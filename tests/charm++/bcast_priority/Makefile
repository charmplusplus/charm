-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: bcast_priority

bcast_priority: bcast_priority.decl.h bcast_priority.C
	$(CHARMC) -language charm++ bcast_priority.C -o bcast_priority

bcast_priority.decl.h: bcast_priority.ci
	$(CHARMC) bcast_priority.ci

clean:
	rm -f *.decl.h *.def.h *.o bcast_priority charmrun

test: all
	$(call run, ./bcast_priority +p1)
	$(call run, ./bcast_priority +p2)
	$(call run, ./bcast_priority +p4)

testp: all
	$(call run, ./bcast_priority +p$(P))

smptest: all
	$(call run, ./bcast_priority +p2 ++ppn 2)
	$(call run, ./bcast_priority +p4 ++ppn 2)
