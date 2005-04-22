
FFTW_HOME=$(HOME)/fftw

CHARMC=$(CDIR)/bin/charmc $(OPTS) $(FLAGS)

CDIR=../../../..
LIBDIR=$(CDIR)/lib
FFTW_LIBDIR=$(FFTW_HOME)/lib

INCLUDE=-I$(FFTW_HOME)/include
COMPILER=$(CHARMC) $(INCLUDE)

LIB=libmoduleckPairCalculator.a
LIBOBJ=ckPairCalculator.o pairCalculator.o 

HEADERS= $(CDIR)/include/pairCalculator.h\
	 $(CDIR)/include/pairutil.h \
         $(CDIR)/include/ckPairCalculator.h\
         $(CDIR)/include/ckPairCalculator.decl.h\
         $(CDIR)/include/ckPairCalculator.def.h\

all: $(LIB)  $(HEADERS)

$(HEADERS): pairCalculator.h ckPairCalculator.h ckPairCalculator.decl.h 
	/bin/cp pairCalculator.h $(CDIR)/include
	/bin/cp ckPairCalculator.h $(CDIR)/include
	/bin/cp pairutil.h $(CDIR)/include
	/bin/cp ckPairCalculator.decl.h $(CDIR)/include
	/bin/cp ckPairCalculator.def.h $(CDIR)/include

$(LIB): pairCalculator.o ckPairCalculator.o 
	$(COMPILER) -o $(LIB) $(LIBOBJ) -module commlib -module CkSparseContiguousReducer 
	/bin/cp $(LIB) $(CDIR)/lib/

ckPairCalculator.o: ckPairCalculator.C ckPairCalculator.h ckPairCalculator.decl.h 
	$(COMPILER)  ckPairCalculator.C

pairCalculator.o: pairCalculator.C pairCalculator.h ckPairCalculator.h ckPairCalculator.decl.h
	$(COMPILER)  pairCalculator.C

ckPairCalculator.decl.h: ckPairCalculator.ci
	$(CHARMC) ckPairCalculator.ci 

myTest: myTest.o $(all)
	$(COMPILER) -o myTest myTest.o -module ckPairCalculator -module CkSparseContiguousReducer

myTest.o: myTest.C myTest.decl.h
	$(COMPILER) myTest.C

myTest.decl.h: myTest.ci
	$(COMPILER) myTest.ci

clean:
	rm -f core 
	rm -f libmoduleckPairCalculator.a *.o charmrun
	rm -f *.decl.h *.def.h
	rm -f *~
