include ../common.mk

LIB = libmoduleCkCache.a
LIBOBJ = CkCache.o

HEADERS = $(CDIR)/include/CkCache.decl.h \
          $(CDIR)/include/CkCache.def.h \
          $(CDIR)/include/CkCache.h 
LIBDEST =  $(LIBDIR)/$(LIB)

CIFILES = CkCache.ci

all: $(LIBDEST) $(HEADERS)

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ) 

CkCache.def.h CkCache.decl.h: INTERFACE

INTERFACE: $(CIFILES)
	$(CHARMC) -E -c CkCache.ci
	touch INTERFACE

CkCache.o: CkCache.C $(HEADERS)
	$(CHARMC) -I../../.. -c -o CkCache.o CkCache.C

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB) INTERFACE

realclean: clean
	rm -f $(LIBDEST) $(HEADERS)
