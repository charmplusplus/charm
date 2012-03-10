CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=CkIO
LIB = $(CDIR)/lib/libmodule$(MODULE).a
LIBOBJ = ckio.o

HEADERS = ckio.h $(MODULE).decl.h $(MODULE).def.h

all: $(LIBDEST)$(LIB)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

ckio.o: ckio.C headers
	$(CHARMC) -c $<

$(MODULE).def.h: $(MODULE).decl.h

$(MODULE).decl.h: ckio.ci
	$(CHARMC) -c $<

clean:
	rm -f *.o *.decl.h *.def.h $(LIB) headers
