CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=ampi.h
HEADDEP=$(HEADERS) ampiimpl.h ampi.decl.h
COMPAT=compat_ampius.o compat_ampifus.o
OBJS=ampi.o ampif.o compat_ampius.o
OBJSF=ampi.o ampif.o compat_ampifus.o

DEST=$(CDIR)/lib/libmoduleampi.a
DESTF=$(CDIR)/lib/libmoduleampif.a

all: AMPI AMPIF

AMPI: $(DEST) 

AMPIF: $(DESTF)

$(DEST): $(OBJS) headers
	$(CHARMC) -DMPI_FORTRAN=0 -c ampisetup.C
	$(CHARMC) $(OBJS) ampisetup.o -o $@

$(DESTF): $(OBJSF) headers
	$(CHARMC) -DMPI_FORTRAN=1 -c ampisetup.C
	-$(CHARMC) -c ampimod.f90
	-$(CHARMC) -cpmod ../../../../include ampi.M
	-$(CHARMC) $(OBJSF) ampimod.o ampisetup.o -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

$(COMPAT):
	@for o in $(COMPAT); \
	do \
		file=`echo $$o | sed -e "s/\.o/.c/g"`; \
		echo "$(CHARMC) -c $$file..."; \
		$(CHARMC) -c $$file || exit 1; \
	done

ampi.o: ampi.C $(HEADDEP)
	$(CHARMC) -c ampi.C

ampif.o: ampif.C $(HEADDEP)
	$(CHARMC) -c ampif.C

ampi.decl.h ampi.def.h: ampi.ci
	$(CHARMC) ampi.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) $(DESTF) conv-host charmrun headers *.mod
