CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
DEST=$(CDIR)/lib/libmodulebarrier.a

all: $(DEST) $(CDIR)/include/barrier.h $(CDIR)/include/barrier.decl.h

$(DEST): barrier.o
	$(CHARMC) -o $(DEST) barrier.o

barrier.o: barrier.C barrier.h barrier.decl.h barrier.def.h
	$(CHARMC) -c barrier.C

barrier.decl.h barrier.def.h: barrier.ci
	$(CHARMC) barrier.ci

$(CDIR)/include/barrier.decl.h: barrier.decl.h
	/bin/cp barrier.decl.h $(CDIR)/include

$(CDIR)/include/barrier.h: barrier.h
	/bin/cp barrier.h $(CDIR)/include

clean:
	rm -f *.decl.h *.def.h *.o $(DEST)
