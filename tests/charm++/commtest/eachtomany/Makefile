# Makefile for compiling Comlib tests

CHARMC=../../../../bin/charmc $(OPTS)



OBJS = eachtomany.o

all: eachtomany

eachtomany: $(OBJS)
	$(CHARMC) -g -language charm++ -o eachtomany $(OBJS) -module comlib

eachtomany.decl.h: eachtomany.ci
	$(CHARMC)  eachtomany.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o eachtomany charmrun *.log *.sum *.sts *~

eachtomany.o: eachtomany.C eachtomany.decl.h
	$(CHARMC) -c -g eachtomany.C

test: eachtomany
	./charmrun ./eachtomany +p1 $(TESTOPTS)
	./charmrun ./eachtomany +p2  $(TESTOPTS)
	./charmrun ./eachtomany +p4  $(TESTOPTS)


debug: eachtomany
	./charmrun ./eachtomany +p2 ++debug-no-pause

bgtest: all
	echo "Not running bgtest for comlib test prgoram"