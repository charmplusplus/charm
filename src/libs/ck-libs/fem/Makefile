CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

all: libfem.a $(CDIR)/include/fem.h $(CDIR)/include/fem.decl.h

libfem.a: fem.o
	@echo 'building libfem.a...'
	@$(CHARMC) -cp $(CDIR)/lib -o libfem.a fem.o

fem.o: fem.C fem.h fem.decl.h fem.def.h
	@echo '.... fem.C...'
	@$(CHARMC) -c fem.C

fem.decl.h fem.def.h: fem.ci
	@echo '.... fem interface...'
	@$(CHARMC) fem.ci

$(CDIR)/include/fem.decl.h: fem.decl.h
	@/bin/cp fem.decl.h $(CDIR)/include

$(CDIR)/include/fem.h: fem.h
	@/bin/cp fem.h $(CDIR)/include

clean:
	rm -rf libfem.a *.def.h *.decl.h *.o SunWS_cache
