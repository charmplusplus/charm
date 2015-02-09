-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = pgm.o

all: pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -module commlib -o pgm $(OBJS)

pgm.decl.h: pgm.ci
	$(CHARMC) pgm.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun

pgm.o: pgm.C pgm.decl.h
	$(CHARMC) -c pgm.C

test: all
	$(call run, ./pgm +p4 )

bgtest: all
	$(call run, ./pgm +p4 +x2 +y2 +z1 )
