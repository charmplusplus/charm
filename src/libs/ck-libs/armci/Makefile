CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

DEST= $(CDIR)/lib/libmodulearmci.a

all: $(DEST)

$(DEST): armci_setup.o armci_vp.o armci_api.o armci_util.o
	$(CHARMC) armci_setup.o armci_vp.o armci_api.o armci_util.o \
		-o $@
	cp armci.h $(CDIR)/include

armci_setup.o: armci_setup.C armci.decl.h armci.def.h armci_impl.h
	$(CHARMC) $(OPTS) -c armci_setup.C

armci_vp.o: armci_vp.C armci.decl.h armci.def.h armci_impl.h
	$(CHARMC) $(OPTS) -c armci_vp.C

armci_api.o: armci_api.C armci.decl.h armci.def.h armci.h \
		armci_impl.h
	$(CHARMC) $(OPTS) -c armci_api.C

armci_util.o: armci_util.C armci_impl.h
	$(CHARMC) $(OPTS) -c armci_util.C

armci.decl.h armci.def.h: armci.ci
	$(CHARMC) armci.ci

clean:
	rm -f *.decl.h *.def.h *.o core

