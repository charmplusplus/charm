SRC=../../../src
CHARMC=../../../bin/charmc $(OPTS)

OBJS = broadcast.o

all:	pgm

projections: pgm.prj

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

pgm.prj: $(OBJS)
	$(CHARMC) -language charm++ -o pgm.prj -tracemode projections $(OBJS)

cifiles: broadcast.ci
	$(CHARMC)  broadcast.ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles pgm.exe pgm.pdb pgm.ilk

broadcast.o: broadcast.C cifiles
	$(CHARMC) -I$(SRC)/conv-core broadcast.C

test: all
	@echo "Inter-processor broadcast.."
	./charmrun ./pgm +p2 $(TESTOPTS)

