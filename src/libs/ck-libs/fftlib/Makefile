FFTW_HOME=$(HOME)/fftw
CHARMC=$(CDIR)/bin/charmc $(OPTS) $(FLAGS)

CDIR=../../../..
LIBDIR=$(CDIR)/lib
FFTW_LIBDIR=$(FFTW_HOME)/lib

OPTS=-g

INCLUDE=-I$(FFTW_HOME)/include
COMPILER=$(CHARMC) $(INCLUDE)

LIB=libmodulefftlib.a
LIBDEST=$(CDIR)/lib/$(LIB)
LIBOBJ=fftlib-normal.o rfftlib-normal.o fftlib-sparse.o translines.o
INCDIR=$(CDIR)/include/fftlib
HEADERS= $(INCDIR)/fftlib.h\
	 $(INCDIR)/fftlib.decl.h\
	 $(INCDIR)/fftlib.def.h

all: $(LIBDEST) $(HEADERS)

$(HEADERS): fftlib.h fftlib.decl.h
	test ! -d $(INCDIR) && mkdir $(INCDIR) || true
	/bin/cp fftlib.h $(INCDIR)
	/bin/cp fftlib.decl.h $(INCDIR)
	/bin/cp fftlib.def.h $(INCDIR)

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ)
	/bin/cp libmodulefftlib.dep $(LIBDIR)


fftlib-normal.o: fftlib-normal.C fftlib.h fftlib.decl.h
	$(COMPILER) fftlib-normal.C

rfftlib-normal.o: rfftlib-normal.C fftlib.h fftlib.decl.h
	$(COMPILER) rfftlib-normal.C

fftlib-sparse.o: fftlib-sparse.C fftlib.h fftlib.decl.h
	$(COMPILER) fftlib-sparse.C

translines.o: translines.C fftlib.decl.h
	$(COMPILER) translines.C

fftlib.def.h: fftlib.decl.h

fftlib.decl.h: fftlib.ci
	$(CHARMC) fftlib.ci

clean:
	rm -f core 
	rm -f $(LIB) $(LIBOBJ)
	rm -f *.decl.h *.def.h util.o
