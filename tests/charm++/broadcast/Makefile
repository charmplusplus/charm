SRC=../../../src
CHARMC=../../../bin/charmc $(OPTS)

OBJS = broadcast.o

all:	pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

cifiles: broadcast.ci
	$(CHARMC)  broadcast.ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles pgm.exe pgm.pdb pgm.ilk

broadcast.o: broadcast.C cifiles
	$(CHARMC) -I$(SRC)/conv-core broadcast.C

test: all
	@echo "Intra-processor broadcast.."
	./charmrun ./pgm +p1 $(TESTOPTS)
	@echo "Inter-processor broadcast.."
	./charmrun ./pgm +p2 $(TESTOPTS)

bgtest: all
	@echo "Intra-processor broadcast.."
	./charmrun ./pgm +p1 +x1 +y1 +z1
	@echo "Inter-processor broadcast.."
	./charmrun ./pgm +p2 +x2 +y1 +z1

