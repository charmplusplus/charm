CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ampi.h ampif.h mpio.h mpiof.h ampiimpl.h ampiProjections.h
HEADDEP=$(HEADERS) ampiimpl.h ampi.decl.h ampiEvents.h
COMPAT=compat_ampius.o compat_ampifus.o compat_ampi.o \
       compat_ampim.o compat_ampifm.o compat_ampicm.o \
	   compat_ampicpp.o
OBJS=ampi.o ampif.o ampiProjections.o ampiOneSided.o \
     ampiMisc.o mpich-alltoall.o  
ROMIOMAKE=romio/adio/ad_hfs/Makefile romio/adio/ad_nfs/Makefile romio/adio/ad_pfs/Makefile romio/adio/ad_piofs/Makefile \
	  romio/adio/ad_pvfs/Makefile romio/adio/ad_sfs/Makefile romio/adio/ad_testfs/Makefile romio/adio/ad_ufs/Makefile \
	  romio/adio/ad_xfs/Makefile romio/adio/common/Makefile romio/adio/include/romioconf.h romio/Makefile \
	  romio/mpi2-other/array/Makefile romio/mpi2-other/info/Makefile romio/mpi-io/Makefile \
	  romio/include/mpio.h romio/include/mpiof.h

LIB=libmoduleampi
LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
DESTF=$(LIBDIR)/$(LIB)f.a
COMPATLIB=$(LIBDIR)/libampi-compat.a
DUMMYROMIOLIB=$(LIBDIR)/libampiromio.a

.SUFFIXES:
.SUFFIXES: .C .o

.C.o:
	$(CHARMC) -c $<

.NOTPARALLEL:

all: AMPI AMPIF $(COMPATLIB) $(DUMMYROMIOLIB)
	-cd $(CDIR)/tmp; CHARMINC=.; . ./conv-config.sh; cd libs/ck-libs/ampi; if test -n "$$CMK_AMPI_WITH_ROMIO" ; then ($(MAKE) AMPI-ROMIO); fi

$(ROMIOMAKE): $(COMPATLIB)
	cd romio && ./config.sh && cd .. 

AMPI-ROMIO: AMPI AMPIF $(ROMIOMAKE)
	cd romio && ./install.sh && cd .. 

AMPI: $(DEST) 
	@echo "AMPI built successfully"

AMPIF: $(DESTF)
	@echo "AMPIF built successfully"

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

$(DESTF): $(OBJS) headers
	-$(CHARMC) -c ampifimpl.f90
	-$(CHARMC) -c ampimod.f90
	-$(CHARMC) -cpmod ../../../../include mpi.M
	-$(CHARMC) $(OBJS) ampifimpl.o ampimod.o -o $@
	-cp $(LIB)f.dep $(LIBDIR)/$(LIB)f.dep

$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

$(DUMMYROMIOLIB): ampiromio.c
	$(CHARMC) ampiromio.c -c ampiromio.o
	$(CHARMC) ampiromio.o -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	cp ampi.h $(CDIR)/include/mpi.h
	cp ampif.h $(CDIR)/include/mpif.h
#	cp mpiCC mpicc mpicxx mpif77 mpif90 $(CDIR)/bin
	cp ampiCC $(CDIR)/bin/ampicc
	cp ampiCC $(CDIR)/bin/ampiCC
	cp ampiCC $(CDIR)/bin/ampicxx
	cp ampif77 $(CDIR)/bin/ampif77
	cp ampif90 $(CDIR)/bin/ampif90
	touch headers

compat_ampius.o: compat_ampius.C

compat_ampifus.o: compat_ampifus.C

compat_ampim.o: compat_ampim.C

compat_ampifm.o: compat_ampifm.C

compat_ampicm.o: compat_ampicm.C

compat_ampi.o: compat_ampi.c
	$(CHARMC) -c compat_ampi.c

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

ampiProjections.o: ampiProjections.C ampiEvents.h ampiProjections.h $(HEADDEP)
	$(CHARMC) -c ampiProjections.C

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) $(DESTF) $(COMPATLIB) conv-host charmrun headers *.mod
	- cd romio && $(MAKE) clean
