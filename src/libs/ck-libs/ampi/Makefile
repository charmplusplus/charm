CDIR := $(shell cd ../../../.. && pwd)
-include $(CDIR)/include/conv-mach-opt.mak
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ampi.h ampif.h mpio.h mpiof.h ampiimpl.h ampi-interoperate.h ampi.decl.h
HEADDEP=$(HEADERS) ampiimpl.h ddt.h \
		../tcharm/tcharm.h ../tcharm/tcharm_impl.h
COMPAT=compat_ampius.o compat_ampifus.o compat_ampi.o \
       compat_ampim.o compat_ampifm.o compat_ampicm.o \
	   compat_ampicpp.o
OBJS=ampi.o ampif.o ampiOneSided.o \
     ampiMisc.o ddt.o mpich-alltoall.o ampi_mpix.o ampi_noimpl.o

AMPI_LIB=libmoduleampi
AMPI_LIBDIR=$(CDIR)/lib
DEST=$(AMPI_LIBDIR)/$(AMPI_LIB).a
DESTF=$(AMPI_LIBDIR)/$(AMPI_LIB)f.a
COMPATLIB=$(AMPI_LIBDIR)/libampi-compat.a
ROMIOLIB=$(AMPI_LIBDIR)/libampiromio

ifeq ($(CMK_NO_BUILD_SHARED),false)
	ROMIOLIBSHARED=$(CDIR)/lib_so/libampiromio.$(CMK_SHARED_SUF)
endif

ifeq ($(CMK_AMPI_WITH_ROMIO),true)
	ROMIO=$(ROMIOLIB).a
endif

.SUFFIXES:
.SUFFIXES: .C .o

.C.o:
	$(CHARMC) -c $<

all: AMPI AMPIF $(COMPATLIB) $(ROMIO)

# AMPI and AMPIF are order-only dependencies (|), i.e., no up-to-date check
# is done, in order to prevent ROMIO recompilation when AMPI changes.
$(ROMIO): | AMPI AMPIF
	@echo "Building ROMIO"
	cd romio && \
		CFLAGS="$(OPTS) -ampi-romio-configure" \
		FFLAGS="$(OPTS) -ampi-romio-configure" \
		F77="$(CDIR)/bin/ampif90" \
		CC="$(CDIR)/bin/ampicc" \
		./configure \
		--disable-aio \
		--enable-f77=`test -f $(DESTF) && echo yes || echo no` \
		--srcdir=`pwd`

	cp romio/include/mpio.h romio/include/mpiof.h $(CDIR)/include
	$(MAKE) -C romio
	cp romio/lib/libmpio.a $@
	if [ -n "$(ROMIOLIBSHARED)" ]; then \
		echo "Building dynamic ROMIO library: $(ROMIOLIBSHARED)"; \
		find romio/ -name \*.o -exec $(CHARMC) -o $(ROMIOLIBSHARED) {} + ; \
	fi
	@echo "ROMIO built successfully"

AMPI: $(DEST)
	@echo "AMPI built successfully"

AMPIF: $(DESTF)
	@echo "AMPIF built successfully"

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@
	cp $(AMPI_LIB).dep $(AMPI_LIBDIR)/$(AMPI_LIB).dep

$(DESTF): $(OBJS) headers
	-$(CHARMC) -c ampifimpl.f90
	-$(CHARMC) -c ampimod.f90
	-$(CHARMC) -cpmod ../../../../include mpi.M
	-$(CHARMC) $(OBJS) ampifimpl.o ampimod.o -o $@
	-cp $(AMPI_LIB)f.dep $(AMPI_LIBDIR)/$(AMPI_LIB)f.dep

$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	cp ampi.h $(CDIR)/include/mpi.h
	cp ampif.h $(CDIR)/include/mpif.h
	cp ampiimpl.h $(CDIR)/include/          # For AMPI + Charm++ interop
	cp ddt.h $(CDIR)/include/               # For AMPI + Charm++ interop
	cp ampi-interoperate.h $(CDIR)/include/ # For AMPI + Charm++ interop
	cp ampiCC $(CDIR)/bin/ampicc
	cp ampiCC $(CDIR)/bin/ampiCC
	cp ampiCC $(CDIR)/bin/ampicxx
	cp ampif77 $(CDIR)/bin/ampif77
	cp ampif90 $(CDIR)/bin/ampif90
	cp ampirun $(CDIR)/bin/ampirun
	touch headers

compat_ampius.o: compat_ampius.C

compat_ampifus.o: compat_ampifus.C

compat_ampim.o: compat_ampim.C

compat_ampifm.o: compat_ampifm.C

compat_ampicm.o: compat_ampicm.C

compat_ampi.o: compat_ampi.c
	$(CHARMC) -c compat_ampi.c

ampi_mpix.o: ampi_mpix.C $(HEADDEP)

ampi_noimpl.o: ampi_noimpl.C $(HEADDEP)

compat_ampicpp.o: compat_ampicpp.C
	$(CHARMC) -c compat_ampicpp.C

ampi.o: ampi.C $(HEADDEP)
	$(CHARMC) -c ampi.C

ampif.o: ampif.C $(HEADDEP)
	$(CHARMC) -c ampif.C

ampiOneSided.o: ampiOneSided.C ampiimpl.h $(HEADDEP)
	$(CHARMC) -c ampiOneSided.C

ampiMisc.o: ampiMisc.C ampiimpl.h $(HEADDEP)
	$(CHARMC) -c ampiMisc.C

mpich-alltoall.o: mpich-alltoall.C $(HEADDEP)
	$(CHARMC) -c mpich-alltoall.C

ampi.decl.h ampi.def.h: ampi.ci
	$(CHARMC) ampi.ci

ddt.o: ddt.C ddt.h $(HEADDEP)

clean:
	rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) $(DESTF) $(COMPATLIB) conv-host charmrun headers *.mod
	- cd romio && $(MAKE) clean
