CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=PythonCCS.h PythonCCS.decl.h PythonCCS.def.h
HEADDEP=$(HEADERS)
OBJS=PythonCCS.o
DEST=$(CDIR)/lib/libmodulePythonCCS.a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT)
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	ln -s $(HEADERS) $(CDIR)/include/
	touch headers

PythonCCS.o: PythonCCS.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) PythonCCS.C

PythonCCS.decl.h PythonCCS.def.h: PythonCCS.ci
	$(CHARMC) PythonCCS.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
