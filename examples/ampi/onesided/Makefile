-include ../../common.mk
OPTS=-O3
CHARMBASE=../../../
CHARMC=../../../bin/charmc -language ampi $(OPTS)
MPICC=mpicc $(OPTS)
TOKENS=6

all: GetPutTest IgetTest

GetPutTest: GetPutTest.C
	$(CHARMC) GetPutTest.C -language ampi  -o GetPutTest
#	$(MPICC) GetPutTest.C -o GetPutTest-mpi
IgetTest: IgetTest.C
	$(CHARMC) IgetTest.C -language ampi  -o IgetTest

IgetTest-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS) $(OPTS)"; cd -;
	rm IgetTest; make IgetTest OPTS="$(OPTS)";
	
test: all
	$(call run, +p2 ./IgetTest 100000 16 +vp2)

clean:
	rm -rf charmrun conv-host moduleinit* *.o GetPutTest IgetTest GetPutTest-mpi *~ *.sts core

