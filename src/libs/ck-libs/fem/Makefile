CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

all: libfem.a libfemf.a $(CDIR)/include/fem.h $(CDIR)/include/fem.decl.h \
     $(CDIR)/include/femf.h $(CDIR)/lib/fmain.o map fmap gmap fgmap

libfem.a: fem.C fem.h femf.h fem.decl.h fem.def.h map.C
	$(CHARMC) -DFEM_FORTRAN=0 -c fem.C
	$(CHARMC) -DMAP_MAIN=0 -DFEM_FORTRAN=0 -c map.C
	$(CHARMC) -cp $(CDIR)/lib -o libfem.a fem.o map.o

libfemf.a: fem.C fem.h femf.h fem.decl.h fem.def.h map.C
	$(CHARMC) -DFEM_FORTRAN=1 -c fem.C
	$(CHARMC) -DMAP_MAIN=0 -DFEM_FORTRAN=1 -c map.C
	$(CHARMC) -cp $(CDIR)/lib -o libfemf.a fem.o map.o

map: map.C fem.h
	$(CHARMC) -seq -DMAP_MAIN=1 -DFEM_FORTRAN=0 -c map.C
	$(CHARMC) -seq -cp $(CDIR)/bin -o map map.o -lmetis -lm -language c++

fmap: map.C fem.h
	$(CHARMC) -seq -DMAP_MAIN=1 -DFEM_FORTRAN=1 -c map.C
	$(CHARMC) -seq -cp $(CDIR)/bin -o fmap map.o -lmetis -lm -language c++

gmap: map.C fem.h
	$(CHARMC) -seq -DMAP_MAIN=1 -DMAP_GRAPH=1 -DFEM_FORTRAN=0 -c map.C
	$(CHARMC) -seq -cp $(CDIR)/bin -o gmap map.o -lmetis -lm -language c++

fgmap: map.C fem.h
	$(CHARMC) -seq -DMAP_MAIN=1 -DMAP_GRAPH=1 -DFEM_FORTRAN=1 -c map.C
	$(CHARMC) -seq -cp $(CDIR)/bin -o fgmap map.o -lmetis -lm -language c++

fmain.o: fmain.f90
	$(CHARMC) -c fmain.f90

fem.decl.h fem.def.h: fem.ci
	$(CHARMC) fem.ci

$(CDIR)/include/fem.decl.h: fem.decl.h
	/bin/cp fem.decl.h $(CDIR)/include

$(CDIR)/include/fem.h: fem.h
	/bin/cp fem.h $(CDIR)/include

$(CDIR)/include/femf.h: femf.h
	/bin/cp femf.h $(CDIR)/include

$(CDIR)/lib/fmain.o: fmain.o
	/bin/cp fmain.o $(CDIR)/lib

clean:
	rm -rf libfem.a libfemf.a fmain.o *.def.h *.decl.h *.o SunWS_cache
