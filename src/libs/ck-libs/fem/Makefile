CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

all: libfem.a $(CDIR)/include/fem.h $(CDIR)/include/fem.decl.h

libfem.a: fem.o
	$(CHARMC) -cp $(CDIR)/lib -o libfem.a fem.o

fem.o: fem.C fem.h fem.decl.h fem.def.h
	$(CHARMC) -c fem.C

fem.decl.h fem.def.h: fem.ci
	$(CHARMC) fem.ci

$(CDIR)/include/fem.decl.h: fem.decl.h
	cp fem.decl.h $(CDIR)/include

$(CDIR)/include/fem.h: fem.h
	cp fem.h $(CDIR)/include

