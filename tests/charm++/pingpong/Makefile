CHARMC=../../../bin/charmc $(OPTS)

OBJS = pingpong.o

all:	cifiles pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun

pingpong.o: pingpong.C
	$(CHARMC) -c pingpong.C

test: all
	@echo "Intra-processor Pingpong.."
	./charmrun ./pgm +p1
	@echo "Inter-processor Pingpong.."
	./charmrun ./pgm +p2
