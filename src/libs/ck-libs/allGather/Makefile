CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
DEST=$(CDIR)/lib/libmoduleallGather.a

all: $(DEST) $(CDIR)/include/allGather.h $(CDIR)/include/allGather.decl.h $(CDIR)/include/allGather.def.h

$(DEST): allGather.o
	$(CHARMC) -o $(DEST) allGather.o

allGather.o: allGather.C allGather.h allGather.decl.h allGather.def.h
	$(CHARMC) -c allGather.C

allGather.decl.h allGather.def.h: allGather.ci
	$(CHARMC) allGather.ci

$(CDIR)/include/allGather.decl.h: allGather.decl.h
	/bin/cp allGather.decl.h $(CDIR)/include

$(CDIR)/include/allGather.h: allGather.h
	/bin/cp allGather.h $(CDIR)/include

$(CDIR)/include/allGather.def.h: allGather.def.h
	/bin/cp allGather.def.h $(CDIR)/include

clean:
	rm -f *.decl.h *.def.h *.o $(DEST)
