-include ../../common.mk
OPTS=-O3 
CHARMC=../../../bin/charmc -language ampi $(OPTS)
MPICC=mpicc $(OPTS)

#all : pingpong-2way.c
#	$(CHARMC) pingpong-2way.c -o pgm
#	$(MPICC) pingpong-2way.c -o pgm-mpi

all : pingpong-1way.c
	$(CHARMC) pingpong-1way.c -o pgm
	$(MPICC) pingpong-1way.c -o pgm-mpi

pipe: pingpong-pipeline.c
	$(CHARMC) pingpong-pipeline.c -o pgm-pipe
	$(MPICC) pingpong-pipeline.c -o pgm-pipe-mpi

test: all
	$(call run, +p2 ./pgm 100000 16 +vp2)

clean:
	rm -rf charmrun conv-host moduleinit* *.o pingpong pgm pgm-mpi pgm-pipe pgm-pipe-mpi *~ *.sts core

