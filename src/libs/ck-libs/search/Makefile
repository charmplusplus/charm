CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=search.h problem.h serialtree.h idastar.h search.decl.h
# Not only must the headers be present, they must be in the right place!
HEADDEP=headers search.def.h
OBJS=search.o problem.o serialtree.o idastar.o
DEST=$(CDIR)/lib/libmodulesearch.a
OPTS=-DRANDOM_STARTING_PROC

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	test -d $(CDIR)/include/cklibs || mkdir $(CDIR)/include/cklibs/
	cp $(HEADERS) $(CDIR)/include/cklibs/
	touch headers

search.o: search.C $(HEADDEP)
	$(CHARMC) -c search.C

problem.o: problem.C $(HEADDEP) 
	$(CHARMC) -c problem.C

serialtree.o: serialtree.C $(HEADDEP)
	$(CHARMC) -c serialtree.C

idastar.o: idastar.C $(HEADDEP)
	$(CHARMC) -c idastar.C

search.decl.h search.def.h: search.ci.stamp
	touch $@

search.ci.stamp: search.ci
	$(CHARMC) $<
	touch $@

clean:
	rm -rf *.a *.def.h *.decl.h *.o *.stamp SunWS_cache $(DEST) headers
