CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

all: fem femf

fem: libfem.a $(CDIR)/include/fem.h $(CDIR)/include/fem.decl.h

femf: libfemf.a $(CDIR)/include/femf.h $(CDIR)/lib/fmain.o

libfem.a: fem.C fem.h femf.h fem.decl.h fem.def.h
	$(CHARMC) -DFEM_FORTRAN=0 -c fem.C
	$(CHARMC) -cp $(CDIR)/lib -o libfem.a fem.o

libfemf.a: fem.C fem.h femf.h fem.decl.h fem.def.h
	$(CHARMC) -DFEM_FORTRAN=1 -c fem.C
	$(CHARMC) -cp $(CDIR)/lib -o libfemf.a fem.o

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
