CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=liveViz3d.h liveViz3d.decl.h ckviewable.h ckviewpoint.h
HEADDEP=$(HEADERS) 
OBJS=ckviewpoint.o liveViz3d.o ckviewable.o
DEST=$(CDIR)/lib/libmoduleliveViz3d.a
OBJS2=ckviewpoint.o ckviewable.o
DEST2=$(CDIR)/lib/libliveViz3d-client.a

all: $(DEST) $(DEST2)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

$(DEST2): $(OBJS2) $(COMPAT) headers
	$(CHARMC) $(OBJS2) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

liveViz3d.o: liveViz3d.C $(HEADDEP)
	$(CHARMC) -c liveViz3d.C

ckviewable.o: ckviewable.cpp $(HEADDEP)
	$(CHARMC) -c ckviewable.cpp

ckviewpoint.o: ckviewpoint.cpp $(HEADDEP)
	$(CHARMC) -c ckviewpoint.cpp

liveViz3d.decl.h liveViz3d.def.h: liveViz3d.ci
	$(CHARMC) liveViz3d.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
