FFTW_HOME=/home/net/yanshi/work/cpaimd/fftw
CHARMC=$(CDIR)/bin/charmc $(OPTS) $(FLAGS)

CDIR=../../../..
LIBDIR=$(CDIR)/lib
FFTW_LIBDIR=$(FFTW_HOME)/lib

OPTS=-g
FLAGS=-Wno-deprecated

INCLUDE=-I$(FFTW_HOME)/include
COMPILER=$(CHARMC) $(INCLUDE)

LIB=libmoduleckPairCalculator.a
LIBDEST=$(CDIR)/lib/$(LIB)
LIBOBJ=ckPairCalculator.o pairCalculator.o 
HEADERS= $(CDIR)/include/pairCalculator.h\
         $(CDIR)/include/ckPairCalculator.h\
         $(CDIR)/include/ckPairCalculator.decl.h\
         $(CDIR)/include/ckPairCalculator.def.h\


all:$(LIBDEST)  $(HEADERS)

$(HEADERS): pairCalculator.h ckPairCalculator.h ckPairCalculator.decl.h 
	/bin/cp pairCalculator.h $(CDIR)/include
	/bin/cp ckPairCalculator.h $(CDIR)/include
	/bin/cp ckPairCalculator.decl.h $(CDIR)/include
	/bin/cp ckPairCalculator.def.h $(CDIR)/include

$(LIBDEST): $(LIBOBJ)
	rm -f $(LIB); ar -qc $(LIBDEST) $(LIBOBJ)

libmoduleckPairCalculator.a: pairCalculator.o ckPairCalculator.o 
	$(COMPILER) -o libmoduleckPairCalculator.a pairCalculator.o ckPairCalculator.o -module CkSparseContiguousReducer 
	/bin/cp libmoduleckPairCalculator.a $(CDIR)/lib/.

ckPairCalculator.o: ckPairCalculator.C ckPairCalculator.h ckPairCalculator.decl.h 
	$(COMPILER)  ckPairCalculator.C

pairCalculator.o: pairCalculator.C pairCalculator.h ckPairCalculator.h ckPairCalculator.decl.h
	$(COMPILER)  pairCalculator.C

ckPairCalculator.decl.h: ckPairCalculator.ci
	$(CHARMC) $<

myTest: myTest.o $(all)
	$(COMPILER) -o myTest myTest.o -module ckPairCalculator -module CkSparseContiguousReducer

myTest.o: myTest.C myTest.decl.h
	$(COMPILER) $<

myTest.decl.h: myTest.ci
	$(COMPILER) $<

clean:
	rm -f core 
	rm -f libmoduleckPairCalculator.a *.o charmrun
	rm -f *.decl.h *.def.h
	rm -f *~
