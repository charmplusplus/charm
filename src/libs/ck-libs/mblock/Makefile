CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=patch.h vector3d.h gridutil.h mblock.h mblock_impl.h \
        mblock.decl.h mblock.def.h

all: libmblock.a libmblockf.a $(CDIR)/include/mblock.h \
     $(CDIR)/include/mblockf.h $(CDIR)/lib/fmain.o

libmblock.a: mblock.C patch.C $(HEADERS)
	$(CHARMC) -DMBLK_FORTRAN=0 -c mblock.C
	$(CHARMC) -DMBLK_FORTRAN=0 -c patch.C
	$(CHARMC) -cp $(CDIR)/lib -o libmblock.a mblock.o patch.o $(OBJS)

libmblockf.a: mblock.C patch.C $(HEADERS)
	$(CHARMC) -DMBLK_FORTRAN=1 -c mblock.C
	$(CHARMC) -DMBLK_FORTRAN=1 -c patch.C
	$(CHARMC) -cp $(CDIR)/lib -o libmblockf.a mblock.o patch.o $(OBJS)

fmain.o: fmain.f90
	$(CHARMC) -c fmain.f90

mblock.decl.h mblock.def.h: mblock.ci
	$(CHARMC) mblock.ci

$(CDIR)/include/mblock.h: mblock.h
	/bin/cp mblock.h $(CDIR)/include

$(CDIR)/include/mblockf.h: mblockf.h
	/bin/cp mblockf.h $(CDIR)/include

$(CDIR)/lib/fmain.o: fmain.o
	/bin/cp fmain.o $(CDIR)/lib

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache
