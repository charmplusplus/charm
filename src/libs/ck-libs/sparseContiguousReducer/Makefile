CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libmoduleCkSparseContiguousReducer.a
LIBOBJ = cksparsecontiguousreducer.o

HEADERS = $(CDIR)/include/CkSparseContiguousReducer.decl.h \
          $(CDIR)/include/cksparsecontiguousreducer.h
LIBDEST =  $(CDIR)/lib/$(LIB)

all: $(LIBDEST) $(HEADERS)

$(HEADERS): CkSparseContiguousReducer.decl.h cksparsecontiguousreducer.h
	/bin/cp CkSparseContiguousReducer.decl.h $(CDIR)/include
	/bin/cp CkSparseContiguousReducer.def.h $(CDIR)/include
	/bin/cp cksparsecontiguousreducer.h $(CDIR)/include

$(LIBDEST) : $(LIB)
	/bin/cp $(LIB) $(LIBDEST)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ)

cksparsecontiguousreducer.o: cksparsecontiguousreducer.C cksparsecontiguousreducer.h CkSparseContiguousReducer.decl.h CkSparseContiguousReducer.def.h
	$(CHARMC) -c cksparsecontiguousreducer.C

CkSparseContiguousReducer.decl.h: cksparsecontiguousreducer.ci
	$(CHARMC) -c cksparsecontiguousreducer.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB)
