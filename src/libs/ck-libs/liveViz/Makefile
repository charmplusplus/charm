CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=liveViz.h liveViz.decl.h liveVizPoll.decl.h liveViz0.h colorScale.h 
HEADDEP=$(HEADERS) liveViz.def.h networkVar.h \
	point.h image.h imagelist.h imagenode.h xSortedImageList.h
OBJS=liveViz0.o liveViz.o liveVizPoll.o networkVar.o colorScale.o point.o image.o imagenode.o imagelist.o xSortedImageList.o
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

liveVizPoll.o: liveVizPoll.C $(HEADDEP)
	$(CHARMC) -c liveVizPoll.C

networkVar.o: networkVar.C $(HEADDEP)
	$(CHARMC) -c networkVar.C

colorScale.o: colorScale.C $(HEADDEP)
	$(CHARMC) -c colorScale.C

point.o : point.C $(HEADDEP)
	$(CHARMC) -c point.C

image.o : image.C $(HEADDEP)
	$(CHARMC) -c image.C

imagenode.o : imagenode.C $(HEADDEP)
	$(CHARMC) -c imagenode.C

imagelist.o : imagelist.C $(HEADDEP)
	$(CHARMC) -c imagelist.C

xSortedImageList.o : xSortedImageList.C $(HEADDEP)
	$(CHARMC) -c xSortedImageList.C

liveViz.decl.h liveViz.def.h: liveViz.ci
	$(CHARMC) liveViz.ci

liveVizPoll.decl.h liveVizPoll.def.h: liveVizPoll.ci
	$(CHARMC) liveVizPoll.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
