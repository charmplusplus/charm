FFTW_HOME=/home/net/yanshi/work/cpaimd/fftw
CHARMC=$(CDIR)/bin/charmc $(OPTS) $(FLAGS)

CDIR=../../../..
LIBDIR=$(CDIR)/lib
FFTW_LIBDIR=$(FFTW_HOME)/lib

OPTS=-g
FLAGS=-Wno-deprecated

INCLUDE=-I$(FFTW_HOME)/include
COMPILER=$(CHARMC) $(INCLUDE)

LIB=libfftlib.a
LIBDEST=$(CDIR)/lib/$(LIB)
LIBOBJ=fftlib-normal.o fftlib-sparse.o
HEADERS= $(CDIR)/include/fftlib.h\
	 $(CDIR)/include/fftlib.decl.h\
	 $(CDIR)/include/fftlib.def.h\
	 $(CDIR)/include/util.h\

all: $(LIBDEST) $(HEADERS)

$(HEADERS): fftlib.h fftlib.decl.h
	/bin/cp fftlib.h $(CDIR)/include
	/bin/cp fftlib.decl.h $(CDIR)/include
	/bin/cp fftlib.def.h $(CDIR)/include
	/bin/cp util.h $(CDIR)/include

$(LIBDEST): $(LIBOBJ)
	rm -f $(LIB); ar -qc $(LIBDEST) $(LIBOBJ)


fftlib-normal.o: fftlib-normal.C fftlib.h fftlib.decl.h
	$(COMPILER) fftlib-normal.C

fftlib-sparse.o: fftlib-sparse.C fftlib.h fftlib.decl.h
	$(COMPILER) fftlib-sparse.C

fftlib.decl.h: fftlib.ci
	$(CHARMC) fftlib.ci

clean:
	rm -f core 
	rm -f $(LIB) $(LIBOBJ)
	rm -f *.decl.h *.def.h util.o
