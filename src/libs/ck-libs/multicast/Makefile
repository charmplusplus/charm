CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libckmulticast.a
LIBOBJ = ckmulticast.o

HEADERS = $(CDIR)/include/CkMulticast.decl.h \
          $(CDIR)/include/ckmulticast.h 
LIBDEST =  $(CDIR)/lib/libckmulticast.a

all: $(LIBDEST) $(HEADERS)

$(HEADERS): CkMulticast.decl.h ckmulticast.h 
	/bin/cp CkMulticast.decl.h $(CDIR)/include
	/bin/cp CkMulticast.def.h $(CDIR)/include
	/bin/cp ckmulticast.h $(CDIR)/include

$(LIBDEST) : $(LIB)
	/bin/cp $(LIB) $(LIBDEST)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

ckmulticast.o: ckmulticast.C ckmulticast.h CkMulticast.decl.h CkMulticast.def.h
	$(CHARMC) -c ckmulticast.C

CkMulticast.decl.h: ckmulticast.ci
	$(CHARMC) -c ckmulticast.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB)
