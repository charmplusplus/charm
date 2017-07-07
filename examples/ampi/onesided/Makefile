CHARMRUN=./charmrun
-include ../../common.mk
OPTS=-O3
CHARMBASE=../../../
CHARMC=../../../bin/ampicxx $(OPTS)
MPICC=mpicc $(OPTS)
TOKENS=6

all: GetPutTest IgetTest

GetPutTest: GetPutTest.C
	$(CHARMC) GetPutTest.C -o GetPutTest
#	$(MPICC) GetPutTest.C -o GetPutTest-mpi
IgetTest: IgetTest.C
	$(CHARMC) IgetTest.C -o IgetTest

IgetTest-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS) $(OPTS)"; cd -;
	rm IgetTest; make IgetTest OPTS="$(OPTS)";
	
test-local: all
	$(call run, ./IgetTest +p2 100000 16 +vp2)

clean:
	rm -rf charmrun conv-host moduleinit* *.o GetPutTest IgetTest GetPutTest-mpi *~ *.sts core

