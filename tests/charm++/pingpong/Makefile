SRC=../../../src
CHARMC=../../../bin/charmc $(OPTS)

OBJS = pingpong.o

all:	cifiles pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -pg -o pgm $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun

pingpong.o: pingpong.C
<<<<<<< Makefile
	$(CHARMC) -pg -c pingpong.C
=======
	$(CHARMC) -I$(SRC)/conv-core pingpong.C
>>>>>>> 1.4

test: all
	@echo "Intra-processor Pingpong.."
	./charmrun ./pgm +p1 $(TESTOPTS)
	@echo "Inter-processor Pingpong.."
	./charmrun ./pgm +p2 $(TESTOPTS)

bgtest: all
	@echo "Intra-processor Pingpong.."
	./charmrun ./pgm +p1 +x1 +y1 +z1
	@echo "Inter-processor Pingpong.."
	./charmrun ./pgm +p2 +x2 +y1 +z1

