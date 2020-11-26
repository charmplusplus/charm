-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = pgm.o

all: multicast

multicast: $(OBJS)
	$(CHARMC) -language charm++ -module commlib -o multicast $(OBJS)

pgm.decl.h: pgm.ci
	$(CHARMC) pgm.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o multicast charmrun

pgm.o: pgm.C pgm.decl.h
	$(CHARMC) -c pgm.C

test: all
	$(call run, ./multicast +p4 )
