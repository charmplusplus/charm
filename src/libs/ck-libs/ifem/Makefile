CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ilsi.h  ilsic.h ilsif.h  ifemc.h ifemf.h
HEADDEP=$(HEADERS) ilsi_vec.h
S_OBJS=ilsi.o cg.o
OBJS=$(S_OBJS) ifem.o
LIB=libmoduleifem

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a

all: $(DEST)

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

ifem.o: ifem.C $(HEADDEP)
	$(CHARMC) -c ifem.C

ilsi.o: ilsi.C $(HEADDEP)
	$(CHARMC) -c ilsi.C

cg.o: cg.C $(HEADDEP)
	$(CHARMC) -c cg.C

solver_test: solver_test.C $(HEADDEP) $(S_OBJS)
	$(CHARMC) -seq solver_test.C $(S_OBJS) -o solver_test -lm

test: solver_test
	./solver_test 100

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) solver_test headers
