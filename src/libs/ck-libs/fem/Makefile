CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

OBJS=partition.o map.o
HEADERS=fem.h fem_impl.h fem.decl.h fem.def.h

all: libfem.a libfemf.a $(CDIR)/include/fem.h \
     $(CDIR)/include/femf.h $(CDIR)/lib/fmain.o

libfem.a: fem.C $(HEADERS) $(OBJS)
	$(CHARMC) -DFEM_FORTRAN=0 -c fem.C
	$(CHARMC) -cp $(CDIR)/lib -o libfem.a fem.o $(OBJS)

libfemf.a: fem.C $(HEADERS) $(OBJS)
	$(CHARMC) -DFEM_FORTRAN=1 -c fem.C
	$(CHARMC) -cp $(CDIR)/lib -o libfemf.a fem.o $(OBJS)

fmain.o: fmain.f90
	$(CHARMC) -c fmain.f90

partition.o: partition.C $(HEADERS)
	$(CHARMC) -c partition.C

map.o: map.C $(HEADERS)
	$(CHARMC) -c map.C

fem.decl.h fem.def.h: fem.ci
	$(CHARMC) fem.ci

$(CDIR)/include/fem.h: fem.h
	/bin/cp fem.h $(CDIR)/include

$(CDIR)/include/femf.h: femf.h
	/bin/cp femf.h $(CDIR)/include

$(CDIR)/lib/fmain.o: fmain.o
	/bin/cp fmain.o $(CDIR)/lib

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache
