-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
OPTS=-O3
MPIOPTS=-O3
CHARMC=../../../bin/ampicxx $(OPTS)
MPICC=mpicc $(MPIOPTS) # Should use 'cc' instead of 'mpicc' on Cray systems

#all : pingpong-2way.c
#	$(CHARMC) pingpong-2way.c -o pingpong
#	$(MPICC) pingpong-2way.c -o pingpong-mpi

all : pingpong-1way.c
	$(CHARMC) pingpong-1way.c -o pingpong
#	$(MPICC) pingpong-1way.c -o pingpong-mpi

pipe: pingpong-pipeline.c
	$(CHARMC) pingpong-pipeline.c -o pingpong-pipe
	$(MPICC) pingpong-pipeline.c -o pingpong-pipe-mpi

test: all
	$(call run, +p1 ./pingpong 16 1048576 100 10 1 +vp2)
	$(call run, +p2 ./pingpong 16 1048576 100 10 1 +vp2)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p1 ./pingpong 16 1048576 100 10 1 +vp2 +noCMAForZC)
	$(call run, +p2 ./pingpong 16 1048576 100 10 1 +vp2 +noCMAForZC)
endif

test-bench: all
	$(call run, +p1 ./pingpong 16 67108864 10000 1000 0 +vp2)
	$(call run, +p2 ./pingpong 16 67108674 10000 1000 0 +vp2)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p1 ./pingpong 16 67108864 10000 1000 0 +vp2 +noCMAForZC)
	$(call run, +p2 ./pingpong 16 67108674 10000 1000 0 +vp2 +noCMAForZC)
endif

clean:
	rm -rf charmrun conv-host moduleinit* *.o pingpong pingpong pingpong-mpi pingpong-pipe pingpong-pipe-mpi *~ *.sts core ampirun

