CDIR=../../../..
# The "-msse" is for my stupid SSE code in ckviewpoint.
CHARMC=$(CDIR)/bin/charmc $(OPTS) -msse
DEST=$(CDIR)/lib/libmoduleliveViz3d.a

OPTS=-O

all: $(DEST) 

include Makefile.client

HEADERS=lv3d0.h lv3d0_server.h lv3d1_server.h \
	liveViz3d.decl.h lv3d0.decl.h lv3d1.decl.h ckviewable.h ckviewpoint.h
HEADDEP=$(HEADERS) 
OBJS=$(LV3D_CLIENT_OBJS) lv3d0_server.o lv3d1_server.o

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@
	cp liveViz3d.dep $(CDIR)/lib/libmoduleliveViz3d.dep

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

liveViz3d.o: liveViz3d.C $(HEADDEP)
	$(CHARMC) -c liveViz3d.C

lv3d0_server.o: lv3d0_server.C $(HEADDEP)
	$(CHARMC) -c lv3d0_server.C

lv3d0.o: lv3d0.C $(HEADDEP)
	$(CHARMC) -c lv3d0.C

lv3d1_server.o: lv3d1_server.C $(HEADDEP)
	$(CHARMC) -c lv3d1_server.C

ckviewable.o: ckviewable.cpp $(HEADDEP)
	$(CHARMC) -c ckviewable.cpp

ckviewpoint.o: ckviewpoint.cpp $(HEADDEP)
	$(CHARMC) -c ckviewpoint.cpp

stats.o: stats.cpp $(HEADDEP)
	$(CHARMC) -c stats.cpp

ckpointviewable.o: ckpointviewable.cpp $(HEADDEP)
	$(CHARMC) -c ckpointviewable.cpp

liveViz3d.decl.h liveViz3d.def.h: liveViz3d.ci lv3d0.ci lv3d1.ci
	$(CHARMC) liveViz3d.ci lv3d0.ci lv3d1.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
