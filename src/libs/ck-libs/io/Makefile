CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=CkIO
LIB = $(CDIR)/lib/libmodule$(MODULE).a
LIBOBJ = ckio.o

GENHEADERS = $(MODULE).decl.h $(MODULE).def.h $(MODULE)_impl.decl.h $(MODULE)_impl.def.h
HEADERS = ckio.h $(GENHEADERS)

all: $(LIBDEST)$(LIB)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

ckio.o: ckio.C headers
	$(CHARMC) -c $<

$(GENHEADERS): ckio.ci.stamp
%.ci.stamp: %.ci
	$(CHARMC) -c $<
	touch $@

clean:
	rm -f *.o *.decl.h *.def.h $(LIB) headers *.stamp
