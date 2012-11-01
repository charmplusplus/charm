SRC=../../../src
CHARMC=../../../bin/charmc $(OPTS) -O3

OBJS = ping.o

all:	pgm

projections : pgm.prj

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

pgm.prj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o pgm.prj $(OBJS)

cifiles: ping.ci
	$(CHARMC)  ping.ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles pgm.exe pgm.pdb pgm.ilk

ping.o: ping.C cifiles
	$(CHARMC) -I$(SRC)/conv-core ping.C

test: all
	@echo "Intra-processor ping.."
	./charmrun ./pgm +p1 $(TESTOPTS)
	@echo "Inter-processor ping.."
	./charmrun ./pgm +p2 $(TESTOPTS)

bgtest: all
	@echo "Intra-processor ping.."
	./charmrun ./pgm +p1 +x1 +y1 +z1
	@echo "Inter-processor ping.."
	./charmrun ./pgm +p2 +x2 +y1 +z1

