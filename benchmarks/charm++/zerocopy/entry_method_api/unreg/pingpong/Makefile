-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

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
	$(call run, +p1 ./pgm 1024 8192 2 1)
	$(call run, +p2 ./pgm 1024 8192 2 1)

test-bench: all
	$(call run, ./pgm +p1)
	$(call run, ./pgm +p2)

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles
