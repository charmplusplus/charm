-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: bcast

bcast: bcast.C bcast.decl.h
	$(CHARMC) -language charm++ -o bcast bcast.C

bcast.decl.h: bcast.ci
	$(CHARMC) bcast.ci

clean:
	rm -f *.decl.h *.def.h *.o bcast charmrun

test: all
	$(call run, ./bcast 20000 +p1)
	$(call run, ./bcast 20000 +p2)

testp: all
	$(call run, ./bcast 20000 +p$(P))
