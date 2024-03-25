include ../common.mk
-include $(CDIR)/include/conv-mach-opt.mak
OPTS=-g -Wall
CHARMC=$(CDIR)/bin/charmc $(OPTS)
CHARMXI=$(CHARMC) -intrinsic

HEADERS=PythonCCS.h PythonCCS-client.h PythonCCS.decl.h pythonIncludes.h
HEADDEP=$(HEADERS) PythonCCS.def.h charmdebug_python.def.h
OBJS=PythonCCS.o PythonCCS-client.o
OBJS_DEBUG=charmdebug-python.o
DEST=$(CDIR)/lib/libmodulePythonCCS.a
DEST_DEBUG=$(CDIR)/lib/libmodulecharmdebug_python.a

ifneq ($(CMK_BUILD_PYTHON),)
all: conditional
else
all:
endif

conditional: $(DEST) $(DEST_DEBUG) headers

$(DEST): $(OBJS)
	$(CHARMC) $(OBJS) -o $@

$(DEST_DEBUG): $(OBJS_DEBUG)
	$(CHARMC) $(OBJS_DEBUG) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

PythonCCS.o: PythonCCS.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) PythonCCS.C

PythonCCS-client.o: PythonCCS-client.C $(HEADDEP)
	$(CHARMC) -seq -c $(FLAGS) PythonCCS-client.C

PythonCCS.decl.h PythonCCS.def.h: PythonCCS.ci
	$(CHARMXI) PythonCCS.ci

charmdebug-python.o: charmdebug-python.C $(HEADDEP)
	$(CHARMC) -c $(FLAGS) -I. charmdebug-python.C

charmdebug_python.decl.h charmdebug_python.def.h: charmdebug-python.ci
	$(CHARMXI) charmdebug-python.ci

pythonIncludes.h:
	echo "#include \"python$(CMK_BUILD_PYTHON)/Python.h\"" > pythonIncludes.h
	echo "#include \"python$(CMK_BUILD_PYTHON)/compile.h\"" >> pythonIncludes.h
	echo "#include \"python$(CMK_BUILD_PYTHON)/eval.h\"" >> pythonIncludes.h
	echo "#include \"python$(CMK_BUILD_PYTHON)/node.h\"" >> pythonIncludes.h

clean:
	rm -rf *.a *~ *.def.h *.decl.h *.o SunWS_cache $(DEST) $(DEST_DEBUG) pythonIncludes.h headers
