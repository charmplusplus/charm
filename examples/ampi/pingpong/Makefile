OPTS=-O3
CHARMC=../../../bin/charmc -language ampi $(OPTS) 
MPICC=mpicc

all : pingpongmpi.c
	$(CHARMC) pingpongmpi.c -o pgm
	$(MPICC) pingpongmpi.c -o pgm-mpi

pipe: pingpong-pipeline.c
	$(CHARMC) pingpong-pipeline.c -o pgm.pipe
	$(MPICC) pingpong-pipeline.c -o pgm.pipe-mpi

test: all
	./charmrun +p2 ./pgm 100000 16 +vp2

clean:
	rm -rf charmrun conv-host moduleinit* *.o pingpong *~ *.sts core

