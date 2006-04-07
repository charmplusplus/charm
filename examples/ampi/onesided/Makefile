OPTS=-O3 
CHARMC=../../../bin/charmc -language ampi $(OPTS)
MPICC=mpicc $(OPTS)

all : GetPutTest IGetTest

GetPutTest: GetPutTest.C
	$(CHARMC) GetPutTest.C -language ampi  -o GetPutTest
#	$(MPICC) GetPutTest.C -o GetPutTest-mpi
IGetTest: IGetTest.C
	$(CHARMC) IGetTest.C -language ampi  -o IGetTest

test: all
	./charmrun +p2 ./pgm 100000 16 +vp2

clean:
	rm -rf charmrun conv-host moduleinit* *.o GetPutTest IGetTest GetPutTest-mpi *~ *.sts core

