CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ampi.h ampif.h ampiimpl.h ampiProjections.h
HEADDEP=$(HEADERS) ampiimpl.h ampi.decl.h ampiEvents.h
COMPAT=compat_ampius.o compat_ampifus.o \
       compat_ampim.o compat_ampifm.o compat_ampicm.o
OBJS=ampi.o ampif.o ampiProjections.o ampiOneSided.o \
     ampiMisc.o

LIB=libmoduleampi
LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
DESTF=$(LIBDIR)/$(LIB)f.a
COMPATLIB=$(LIBDIR)/libampi-compat.a

.SUFFIXES:
.SUFFIXES: .C .o

.C.o:
	$(CHARMC) -c $<

all: AMPI AMPIF $(COMPATLIB)

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

$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	cp ampi.h $(CDIR)/include/mpi.h
	cp ampif.h $(CDIR)/include/mpif.h
	cp mpiCC mpicc mpif77 mpif90 $(CDIR)/bin
	touch headers

compat_ampius.o: compat_ampius.C

compat_ampifus.o: compat_ampifus.C

compat_ampim.o: compat_ampim.C

compat_ampifm.o: compat_ampifm.C

compat_ampicm.o: compat_ampicm.C

ampi.o: ampi.C $(HEADDEP)
	$(CHARMC) -c ampi.C

ampif.o: ampif.C $(HEADDEP)
	$(CHARMC) -c ampif.C

ampiOneSided.o: ampiOneSided.C ampiimpl.h $(HEADDEP)
	$(CHARMC) -c ampiOneSided.C

ampiMisc.o: ampiMisc.C ampiimpl.h $(HEADDEP)
	$(CHARMC) -c ampiMisc.C

ampi.decl.h ampi.def.h: ampi.ci
	$(CHARMC) ampi.ci

ampiProjections.o: ampiProjections.C ampiEvents.h ampiProjections.h $(HEADDEP)
	$(CHARMC) -c ampiProjections.C

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) $(DESTF) $(COMPATLIB) conv-host charmrun headers *.mod
