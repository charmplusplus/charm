CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=taskGraph.h taskGraph.decl.h
HEADDEP=$(HEADERS) taskGraph.def.h
OBJS=taskGraph.o
DEST=$(CDIR)/lib/libmoduletaskGraph.a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

taskGraph.o: taskGraph.C $(HEADDEP)
	$(CHARMC) -c taskGraph.C

taskGraph.decl.h taskGraph.def.h: taskGraph.ci
	$(CHARMC) taskGraph.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
