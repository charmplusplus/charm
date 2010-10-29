# Makefile for compiling Comlib tests

CHARMC=/expand/home/abhigupta/openatom/charm/bin/charmc $(OPTS)

OBJS = broadcast.o

all: broadcast

broadcast: $(OBJS)
	$(CHARMC) -language charm++ -o broadcast $(OBJS) -module comlib

broadcast.decl.h: broadcast.ci
	$(CHARMC)  broadcast.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o broadcast charmrun *.log *.sum *.sts *~

broadcast.o: broadcast.C broadcast.decl.h
	$(CHARMC) -c -g broadcast.C

test: all
	./charmrun ./broadcast +p4 10

bgtest: all
	./charmrun ./broadcast +p4 10 +x2 +y2 +z2 +cth1 +wth1
