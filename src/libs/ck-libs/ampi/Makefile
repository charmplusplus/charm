CDIR := $(shell cd ../../../.. && pwd)
-include $(CDIR)/include/conv-mach-opt.mak
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ampi.h ampif.h mpio.h mpiof.h ampiimpl.h \
		ampi-interoperate.h ampi.decl.h \
		mpio_globals.h \
		ampi_functions.h mpio_functions.h
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

AMPIFIMPL_OBJ=$(AMPI_LIBDIR)/ampifimpl.o
AMPIMOD_OBJ=$(AMPI_LIBDIR)/ampimod.o

FUNCPTR_SHIM_OBJ=$(AMPI_LIBDIR)/ampi_funcptr_shim.o
FUNCPTR_FORTRAN_OBJ=$(AMPI_LIBDIR)/ampi_funcptr_fortran.o
FUNCPTR_LOADER_OBJ=$(AMPI_LIBDIR)/ampi_funcptr_loader.o
FUNCPTR_PIPGLOBALS_OBJ=$(AMPI_LIBDIR)/ampi_funcptr_pipglobals.o
FUNCPTR_FSGLOBALS_OBJ=$(AMPI_LIBDIR)/ampi_funcptr_fsglobals.o

AMPI_TARGET = $(DEST) $(FUNCPTR_SHIM_OBJ) $(FUNCPTR_LOADER_OBJ)
AMPIF_TARGET = $(DESTF) $(FUNCPTR_FORTRAN_OBJ)

ifeq (1,$(CMK_SUPPORTS_PIPGLOBALS))
  AMPI_TARGET += $(FUNCPTR_PIPGLOBALS_OBJ)
endif
ifeq (1,$(CMK_SUPPORTS_FSGLOBALS))
  AMPI_TARGET += $(FUNCPTR_FSGLOBALS_OBJ)
endif

ifeq ($(CMK_NO_BUILD_SHARED),false)
	ROMIOLIBSHARED=$(CDIR)/lib_so/libampiromio.$(CMK_SHARED_SUF)
endif

ifeq ($(CMK_AMPI_WITH_ROMIO),1)
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
# update timestamps on these files so that ROMIO's Makefile does not run configure a second time
	@touch romio/aclocal.m4 romio/Makefile.in romio/configure romio/config.status
	cd romio &&                                                                  \
		MPI_IMPL="ampi" FROM_MPICH="no" FROM_LAM="no" FROM_OMPI="no"         \
		MPI_LIB=""                                                           \
		CFLAGS="$(OPTS) -ampi-romio-configure"                               \
		FFLAGS="$(OPTS) -ampi-romio-configure"                               \
		F77="$(CDIR)/bin/ampif90"                                            \
		CC="$(CDIR)/bin/ampicc"                                              \
		./configure                                                          \
		--disable-aio                                                        \
		--enable-f77=`test -f $(DESTF) && echo yes || echo no`               \
		--enable-static=yes                                                  \
		--enable-shared=`test -n "$(ROMIOLIBSHARED)" && echo yes || echo no` \
		--disable-weak-symbols                                               \
		--srcdir=`pwd`

	cp romio/include/mpio.h romio/include/mpiof.h romio/include/mpio_functions.h $(CDIR)/include
	$(MAKE) -C romio
	cp romio/.libs/libromio.a $@
ifneq ($(ROMIOLIBSHARED),)
	cp romio/.libs/libromio.$(CMK_SHARED_SUF) $(ROMIOLIBSHARED)
ifeq ($(CMK_MACOSX),1)
	install_name_tool -id $(ROMIOLIBSHARED) $(ROMIOLIBSHARED)
endif
endif
	@echo "ROMIO built successfully"

AMPI: $(AMPI_TARGET)
	@echo "AMPI built successfully"

AMPIF: $(AMPIF_TARGET)
	@echo "AMPIF built successfully"

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@
	cp $(AMPI_LIB).dep $(AMPI_LIBDIR)/$(AMPI_LIB).dep

$(DESTF): $(OBJS) headers
	-$(CHARMC) -c ampifimpl.f90 -o $(AMPIFIMPL_OBJ)
	-$(CHARMC) -c ampimod.f90 -o $(AMPIMOD_OBJ)
	-$(CHARMC) -cpmod ../../../../include mpi.M
	-$(CHARMC) $(OBJS) $(AMPIFIMPL_OBJ) $(AMPIMOD_OBJ) -o $@
	-cp $(AMPI_LIB)f.dep $(AMPI_LIBDIR)/$(AMPI_LIB)f.dep

$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/

	# Put AMPI's mpi.h in a separate directory to support the MPI machine layer and MPI interop
	mkdir -p $(CDIR)/include/ampi/
	cp ampi.h $(CDIR)/include/ampi/mpi.h

	cp ampif.h $(CDIR)/include/mpif.h
	cp ampiimpl.h $(CDIR)/include/          # For AMPI + Charm++ interop
	cp ddt.h $(CDIR)/include/               # For AMPI + Charm++ interop
	cp ampi-interoperate.h $(CDIR)/include/ # For AMPI + Charm++ interop
	cp mpio_globals.h $(CDIR)/include/
	cp ampiCC $(CDIR)/bin/ampicc
	cp ampiCC $(CDIR)/bin/ampiCC
	cp ampiCC $(CDIR)/bin/ampicxx
	cp ampif77 $(CDIR)/bin/ampif77
	cp ampif90 $(CDIR)/bin/ampif90
	cp ampirun $(CDIR)/bin/ampirun
	cp nm_globals.sh $(CDIR)/bin/nm_globals.sh
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

$(FUNCPTR_SHIM_OBJ): ampi_funcptr_shim.C $(HEADDEP)
	$(CHARMC) -ampi-funcptr-shim -c $< -o $@

$(FUNCPTR_FORTRAN_OBJ): ampif.C $(HEADDEP)
	-$(CHARMC) -ampi-funcptr-shim -c $< -o $@

$(FUNCPTR_LOADER_OBJ): ampi_funcptr_loader.C $(HEADDEP)
	$(CHARMC) -c $< -o $@

$(FUNCPTR_FSGLOBALS_OBJ): ampi_funcptr_fsglobals.C $(HEADDEP)
	$(CHARMC) -c $< -o $@

$(FUNCPTR_PIPGLOBALS_OBJ): ampi_funcptr_pipglobals.C $(HEADDEP)
	$(CHARMC) -c $< -o $@

clean:
	rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) $(DESTF) $(COMPATLIB) conv-host charmrun headers *.mod
	- cd romio && $(MAKE) clean
