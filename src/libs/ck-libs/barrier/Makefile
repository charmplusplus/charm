CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

all: libbarrier.a $(CDIR)/include/barrier.h $(CDIR)/include/Barrier.decl.h

libbarrier.a: barrier.o
	$(CHARMC) -cp $(CDIR)/lib -o libbarrier.a barrier.o

barrier.o: barrier.C barrier.h Barrier.decl.h Barrier.def.h
	$(CHARMC) -c barrier.C

Barrier.decl.h Barrier.def.h: barrier.ci
	$(CHARMC) barrier.ci

$(CDIR)/include/Barrier.decl.h: Barrier.decl.h
	/bin/cp Barrier.decl.h $(CDIR)/include

$(CDIR)/include/barrier.h: barrier.h
	/bin/cp barrier.h $(CDIR)/include

