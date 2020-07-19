CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

ARMCI_LIB := libmodulearmci
ARMCI_LIBDIR := $(CDIR)/lib
DEST := $(ARMCI_LIBDIR)/$(ARMCI_LIB).a

all: $(DEST)

$(DEST): armci_vp.o armci_api.o 
	$(CHARMC) armci_vp.o armci_api.o -o $@
	cp $(ARMCI_LIB).dep $(ARMCI_LIBDIR)/$(ARMCI_LIB).dep
	cp armci.h $(CDIR)/include

armci_vp.o: armci_vp.C armci.decl.h armci.def.h armci_impl.h
	$(CHARMC) $(OPTS) -c armci_vp.C

armci_api.o: armci_api.C armci.decl.h armci.def.h armci.h armci_impl.h
	$(CHARMC) $(OPTS) -c armci_api.C

armci.decl.h: armci.def.h

armci.def.h: armci.ci
	$(CHARMC) armci.ci

clean:
	rm -f *.decl.h *.def.h *.o core

