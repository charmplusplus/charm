CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=liveViz3d.h liveViz3d.decl.h liveViz3d_impl.h viewable.h viewpoint.h
HEADDEP=$(HEADERS) 
OBJS=liveViz3d.o viewable.o
DEST=$(CDIR)/lib/libmoduleliveViz3d.a
OBJS2=viewable.o
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

viewable.o: viewable.C $(HEADDEP)
	$(CHARMC) -c viewable.C

liveViz3d.decl.h liveViz3d.def.h: liveViz3d.ci
	$(CHARMC) liveViz3d.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
