-include ../../../../common.mk
CHARMC=../../../../../bin/charmc $(OPTS)

all:	pgm

OBJS = pingpong.o

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci
	touch cifiles

pingpong.o: pingpong.C cifiles
	$(CHARMC) -c pingpong.C

test: all
	@echo "Intra-processor Pingpong.."
	$(call run, ./pgm +p1 )
	@echo "Inter-processor Pingpong.."
	$(call run, ./pgm +p2 )

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles
