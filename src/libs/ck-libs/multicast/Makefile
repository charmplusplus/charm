CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libmoduleCkMulticast.a
LIBOBJ = ckmulticast.o

HEADERS = $(CDIR)/include/CkMulticast.decl.h \
          $(CDIR)/include/ckmulticast.h 
LIBDEST =  $(CDIR)/lib/$(LIB)

all: $(LIBDEST) $(HEADERS)

$(HEADERS): CkMulticast.decl.h ckmulticast.h 
	/bin/cp CkMulticast.decl.h $(CDIR)/include
	/bin/cp CkMulticast.def.h $(CDIR)/include
	/bin/cp ckmulticast.h $(CDIR)/include

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ) 

ckmulticast.o: ckmulticast.C ckmulticast.h CkMulticast.decl.h
	$(CHARMC) -c ckmulticast.C

CkMulticast.def.h: CkMulticast.decl.h

CkMulticast.decl.h : ckmulticast.ci
	$(CHARMC) -c ckmulticast.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB)
