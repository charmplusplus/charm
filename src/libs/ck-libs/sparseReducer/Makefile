CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libmoduleCkSparseReducer.a
LIBOBJ = cksparsereducer.o

HEADERS = $(CDIR)/include/CkSparseReducer.decl.h \
          $(CDIR)/include/cksparsereducer.h
LIBDEST =  $(CDIR)/lib/$(LIB)

all: $(LIBDEST) $(HEADERS)

$(HEADERS): CkSparseReducer.decl.h cksparsereducer.h
	/bin/cp CkSparseReducer.decl.h $(CDIR)/include
	/bin/cp CkSparseReducer.def.h $(CDIR)/include
	/bin/cp cksparsereducer.h $(CDIR)/include

$(LIBDEST) : $(LIB)
	/bin/cp $(LIB) $(LIBDEST)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ)

cksparsereducer.o: cksparsereducer.C cksparsereducer.h CkSparseReducer.decl.h CkSparseReducer.def.h
	$(CHARMC) -c cksparsereducer.C

CkSparseReducer.decl.h: cksparsereducer.ci
	$(CHARMC) -c cksparsereducer.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB)
