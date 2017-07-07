CHARMRUN=./charmrun
-include ../../../common.mk
SRC=../../../src
CHARMC=../../../../bin/charmc $(OPTS) -g

OBJS = ping.o

all:	pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS) -tracemode perfReport

pgm.prj: $(OBJS)
	$(CHARMC) -tracemode projections -language charm++ -o pgm.prj $(OBJS) -tracemode perfReport

cifiles: ping.ci
	$(CHARMC)  ping.ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles pgm.exe pgm.pdb pgm.ilk

ping.o: ping.C cifiles
	$(CHARMC) -c ping.C

test-local: all
	@echo "Intra-processor ping.."
	$(call run, ./pgm +p1 )
	@echo "Inter-processor ping.."
	$(call run, ./pgm +p2 )

bgtest: all
	@echo "Intra-processor ping.."
	$(call run, +p1 ./pgm +x1 +y1 +z1 )
	@echo "Inter-processor ping.."
	$(call run, +p2 ./pgm +x2 +y1 +z1 )

