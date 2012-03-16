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
	./charmrun pgm +p4 $(TESTOPTS)

bgtest: all
	./charmrun pgm +p4 +x2 +y2 +z1 $(TESTOPTS)
