CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=search.h problem.h serialtree.h idastar.h search.decl.h
HEADDEP=$(HEADERS) search.def.h
OBJS=search.o problem.o serialtree.o idastar.o
DEST=$(CDIR)/lib/libmodulesearch.a
OPTS=-DRANDOM_STARTING_PROC

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

search.o: search.C $(HEADDEP)
	$(CHARMC) -c search.C

problem.o: problem.C $(HEADDEP)
	$(CHARMC) -c problem.C

serialtree.o: serialtree.C $(HEADDEP)
	$(CHARMC) -c serialtree.C

idastar.o: idastar.C $(HEADDEP)
	$(CHARMC) -c idastar.C

search.decl.h search.def.h: search.ci
	$(CHARMC) search.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
