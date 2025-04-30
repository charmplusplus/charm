CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
DEST=$(CDIR)/lib/libmoduleallGather.a

all: $(DEST) $(CDIR)/include/allGather.hh $(CDIR)/include/allGather.decl.h $(CDIR)/include/allGather.def.h

$(DEST): allGather.o
	$(CHARMC) -o $(DEST) allGather.o

allGather.o: allGather.cc allGather.hh allGather.decl.h allGather.def.h
	$(CHARMC) -c allGather.cc

allGather.decl.h allGather.def.h: allGather.ci
	$(CHARMC) allGather.ci

$(CDIR)/include/allGather.decl.h: allGather.decl.h
	/bin/cp allGather.decl.h $(CDIR)/include

$(CDIR)/include/allGather.hh: allGather.hh
	/bin/cp allGather.hh $(CDIR)/include

$(CDIR)/include/allGather.def.h: allGather.def.h
	/bin/cp allGather.def.h $(CDIR)/include

clean:
	rm -f *.decl.h *.def.h *.o $(DEST)
