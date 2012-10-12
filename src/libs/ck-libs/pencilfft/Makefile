include ../common.mk

FFTW_HOME=$(HOME)/fftw-bgq

FFTW_LIBDIR=$(FFTW_HOME)/lib

OPTS=-g -O3

INCLUDE=-I$(FFTW_HOME)/include -DFFTW_ENABLE_FLOAT=1 -DUSE_FFTW_DECLS=1
COMPILER=$(CHARMC) $(INCLUDE)

LIB=libmodulePencilFFT.a
LIBDEST=$(CDIR)/lib/$(LIB)
LIBOBJ=pencilfft.o
INCDIR=$(CDIR)/include/pencilfft
HEADERS= $(INCDIR)/pencilfft.h\
	 $(INCDIR)/PencilFFT.decl.h\
	 $(INCDIR)/PencilFFT.def.h \
	 $(INCDIR)/pencil_api.h


all: $(LIBDEST) $(HEADERS)


$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ)
	/bin/cp libmodulepencilfft.dep $(LIBDIR)

$(HEADERS): pencilfft.h PencilFFT.decl.h
	test ! -d $(INCDIR) && mkdir $(INCDIR) || true
	/bin/cp pencilfft.h $(INCDIR)
	/bin/cp pencil_api.h $(INCDIR)
	/bin/cp PencilFFT.decl.h $(INCDIR)
	/bin/cp PencilFFT.def.h $(INCDIR)

pencilfft.o: pencilfft.C PencilFFT.def.h
	$(COMPILER) pencilfft.C

PencilFFT.def.h: PencilFFT.decl.h

PencilFFT.decl.h: pencilfft.ci
	$(CHARMC) pencilfft.ci

clean:
	rm -f core* *~
	rm -f $(LIB) $(LIBOBJ)
	rm -f *.decl.h *.def.h 
