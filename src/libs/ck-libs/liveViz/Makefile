CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS) -I../../..
FLAGS=-DEXTERIOR_BLACK_PIXEL_ELIMINATION

HEADERS=liveViz.h liveViz.decl.h liveVizPoll.decl.h liveViz0.h colorScale.h ImageData.h
HEADDEP=$(HEADERS) liveViz.def.h ImageData.h
OBJS=liveViz0.o liveViz.o liveVizPoll.o \
	colorScale.o ImageData.o compat_float2rgb.o
DEST=$(CDIR)/lib/libmoduleliveViz.a
DEP=$(CDIR)/lib/libmoduleliveViz.dep

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@
	(cd $(CDIR)/tmp; . ./conv-mach-opt.sh && echo "$$CMK_LIBJPEG") > $(DEP)

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

liveViz0.o: liveViz0.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) liveViz0.C

compat_float2rgb.o: compat_float2rgb.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) compat_float2rgb.C
liveViz.o: liveViz.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) liveViz.C

liveVizPoll.o: liveVizPoll.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) liveVizPoll.C

networkVar.o: networkVar.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) networkVar.C

colorScale.o: colorScale.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) colorScale.C

ImageData.o : ImageData.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) ImageData.C

liveViz.decl.h liveViz.def.h: liveViz.ci
	$(CHARMC) liveViz.ci

liveVizPoll.decl.h liveVizPoll.def.h: liveVizPoll.ci
	$(CHARMC) liveVizPoll.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
