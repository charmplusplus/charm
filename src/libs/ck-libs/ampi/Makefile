CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ampi.h ampif.h
HEADDEP=$(HEADERS) ampiimpl.h ampi.decl.h
COMPAT=compat_ampius.o compat_ampifus.o \
       compat_ampim.o compat_ampifm.o compat_ampicm.o
OBJS=ampi.o ampif.o $(COMPAT)

LIB=libmoduleampi
LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
DESTF=$(LIBDIR)/$(LIB)f.a

all: AMPI AMPIF

AMPI: $(DEST) 
	@echo "AMPI built successfully"

AMPIF: $(DESTF)

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

$(DESTF): $(OBJS) headers
	-$(CHARMC) -c ampimod.f90
	-$(CHARMC) -cpmod ../../../../include mpiincl.M
	-$(CHARMC) $(OBJS) ampimod.o -o $@
	-cp $(LIB)f.dep $(LIBDIR)/$(LIB)f.dep

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	cp ampi.h $(CDIR)/include/mpi.h
	cp ampif.h $(CDIR)/include/mpif.h
	cp mpiCC mpicc mpif77 mpif90 $(CDIR)/bin
	touch headers

$(COMPAT):
	@for o in $(COMPAT); \
	do \
		file=`echo $$o | sed -e "s/\.o/.C/g"`; \
		echo "$(CHARMC) -c $$file..."; \
		$(CHARMC) -c $$file || exit 1; \
	done

ampi.o: ampi.C $(HEADDEP)
	$(CHARMC) -c ampi.C

ampif.o: ampif.C $(HEADDEP)
	$(CHARMC) -c ampif.C

ampi.decl.h ampi.def.h: ampi.ci
	$(CHARMC) ampi.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) $(DESTF) conv-host charmrun headers *.mod
