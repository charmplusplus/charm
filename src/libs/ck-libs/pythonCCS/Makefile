CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=PythonCCS.h PythonCCS-client.h PythonCCS.decl.h
HEADDEP=$(HEADERS) PythonCCS.def.h
OBJS=PythonCCS.o PythonCCS-client.o
DEST=$(CDIR)/lib/libmodulePythonCCS.a

all: $(DEST) headers

$(DEST): $(OBJS)
	$(CHARMC) $(OBJS) -o $@

headers: $(HEADERS)
	ln -sf ../tmp/libs/ck-libs/pythonCCS/PythonCCS.h $(CDIR)/include/
	ln -sf ../tmp/libs/ck-libs/pythonCCS/PythonCCS-client.h $(CDIR)/include/
	ln -sf ../tmp/libs/ck-libs/pythonCCS/PythonCCS.decl.h $(CDIR)/include/

PythonCCS.o: PythonCCS.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) PythonCCS.C

PythonCCS-client.o: PythonCCS-client.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) PythonCCS-client.C

PythonCCS.decl.h PythonCCS.def.h: PythonCCS.ci
	$(CHARMC) PythonCCS.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
