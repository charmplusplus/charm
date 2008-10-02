CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libmoduleCkCache.a
LIBOBJ = CkCache.o

HEADERS = $(CDIR)/include/CkCache.decl.h \
          $(CDIR)/include/CkCache.def.h \
          $(CDIR)/include/CkCache.h 
LIBDEST =  $(CDIR)/lib/$(LIB)

CIFILES = CkCache.ci

all: $(LIBDEST) $(HEADERS)

$(CDIR)/include/%.h: %.h
	/bin/cp $< $@

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ) 

CkCache.def.h CkCache.decl.h: INTERFACE

INTERFACE: $(CIFILES)
	$(CHARMC) -c CkCache.ci
	touch INTERFACE

CkCache.o: CkCache.C $(HEADERS)
	$(CHARMC) -c -o CkCache.o CkCache.C

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB)

realclean: clean
	rm -f $(LIBDEST) $(HEADERS)