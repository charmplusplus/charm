CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=liveViz.h liveViz.decl.h liveViz0.h
HEADDEP=$(HEADERS) liveViz.def.h networkVar.h
OBJS=liveViz0.o liveViz.o networkVar.o
DEST=$(CDIR)/lib/libmoduleliveViz.a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

liveViz0.o: liveViz0.C $(HEADDEP)
	$(CHARMC) -c liveViz0.C

liveViz.o: liveViz.C $(HEADDEP)
	$(CHARMC) -c liveViz.C

networkVar.o: networkVar.C $(HEADDEP)
	$(CHARMC) -c networkVar.C

liveViz.decl.h liveViz.def.h: liveViz.ci
	$(CHARMC) liveViz.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
