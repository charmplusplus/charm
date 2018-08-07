include ../common.mk

LIB = libmoduleCkSparseContiguousReducer.a
LIBOBJ = cksparsecontiguousreducer.o

HEADERS = $(CDIR)/include/CkSparseContiguousReducer.decl.h \
          $(CDIR)/include/CkSparseContiguousReducer.def.h \
          $(CDIR)/include/cksparsecontiguousreducer.h

LIBDEST =  $(CDIR)/lib/$(LIB)

all: $(LIBDEST) $(HEADERS)

$(LIBDEST) : $(LIBOBJ)
	$(CHARMC) -o $@ $^

cksparsecontiguousreducer.o: cksparsecontiguousreducer.C cksparsecontiguousreducer.h CkSparseContiguousReducer.decl.h CkSparseContiguousReducer.def.h
	$(CHARMC) -c cksparsecontiguousreducer.C

CkSparseContiguousReducer.decl.h CkSparseContiguousReducer.def.h: INTERFACE

INTERFACE: CkSparseContiguousReducer.ci
	$(CHARMC) -c CkSparseContiguousReducer.ci
	touch $@

clean:
	rm -f conv-host *.o *.decl.h *.def.h core INTERFACE

realclean: clean
	rm -f $(LIBDEST) $(HEADERS)
