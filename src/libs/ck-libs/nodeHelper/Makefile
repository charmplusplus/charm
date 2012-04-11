OPTS= -O3 -lpthread
CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=NodeHelper
LIB = $(CDIR)/lib/libmodule$(MODULE).a
LIBOBJ = NodeHelper.o

HEADERS = NodeHelperAPI.h $(MODULE).decl.h $(MODULE).def.h

all: $(LIBDEST)$(LIB)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/

NodeHelper.o: NodeHelper.C headers
	$(CHARMC) -c $<

$(MODULE).def.h: $(MODULE).decl.h

$(MODULE).decl.h: NodeHelper.ci
	$(CHARMC) -c $<

clean:
	rm -f *.o *.decl.h *.def.h $(LIB) headers
