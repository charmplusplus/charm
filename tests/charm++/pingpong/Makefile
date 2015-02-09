-include ../../common.mk
SRC=../../../src
CHARMC=../../../bin/charmc $(OPTS)

OBJS = pingpong.o

all:	pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

pgm.prj: $(OBJS)
	$(CHARMC) -tracemode projections -language charm++ -o pgm.prj $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles pgm.exe pgm.pdb pgm.ilk

pingpong.o: pingpong.C cifiles
	$(CHARMC) -I$(SRC)/conv-core pingpong.C

test: all
	@echo "Intra-processor Pingpong.."
	$(call run, ./pgm +p1 )
	@echo "Inter-processor Pingpong.."
	$(call run, ./pgm +p2 )

bgtest: all
	@echo "Intra-processor Pingpong.."
	$(call run, ./pgm +p1 +x1 +y1 +z1 )
	@echo "Inter-processor Pingpong.."
	$(call run, ./pgm +p2 +x2 +y1 +z1 )

