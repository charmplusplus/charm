CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libmoduleCkCache.a
LIBOBJ = CkCache.o

HEADERS = $(CDIR)/include/CkCache.decl.h \
          $(CDIR)/include/CkCache.h 
LIBDEST =  $(CDIR)/lib/$(LIB)

CIFILES = CkCache.ci

all: $(LIBDEST)

$(HEADERS): CkCache.decl.h CkCache.h 
	/bin/cp CkCache.decl.h $(CDIR)/include
	/bin/cp CkCache.h $(CDIR)/include

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ) 

CkCache.def.h: CkCache.decl.h

CkCache.decl.h : CkCache.ci $(CDIR)/bin/charmxi
	$(CHARMC) -c CkCache.ci

CkCache.o: CkCache.C $(HEADERS)
	$(CHARMC) -c -o CkCache.o CkCache.C

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB)

