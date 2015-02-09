-include ../../common.mk
OPTS=-O3 
CHARMBASE=../../../
CHARMC=../../../bin/charmc -language ampi $(OPTS)
MPICC=mpicc $(OPTS)
TOKENS=6

all: GetPutTest IGetTest

GetPutTest: GetPutTest.C
	$(CHARMC) GetPutTest.C -language ampi  -o GetPutTest
#	$(MPICC) GetPutTest.C -o GetPutTest-mpi
IGetTest: IGetTest.C
	$(CHARMC) IGetTest.C -language ampi  -o IGetTest

IGetTest-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS) $(OPTS)"; cd -;
	rm IGetTest; make IGetTest OPTS="$(OPTS)";
	
test: all
	$(call run, +p2 ./pgm 100000 16 +vp2)

clean:
	rm -rf charmrun conv-host moduleinit* *.o GetPutTest IGetTest GetPutTest-mpi *~ *.sts core

