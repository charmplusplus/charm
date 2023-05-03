-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
OPTS=-O3
MPIOPTS=-O3
CHARMC=../../../bin/ampicxx $(OPTS)
MPICC=mpicc $(MPIOPTS) # Should use 'cc' instead of 'mpicc' on Cray systems

all: pingpong

pingpong: pingpong-1way.c
	$(CHARMC) $^ -o $@
#	$(MPICC) $^ -o $@-mpi

pongpong-2way: pingpong-2way.c
	$(CHARMC) $^ -o $@
#	$(MPICC) $^ -o $@-mpi

pingpong-pipeline: pingpong-pipeline.c
	$(CHARMC) $^ -o $@
#	$(MPICC) $^ -o $@-mpi

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
	rm -rf charmrun conv-host moduleinit* *.o pingpong pingpong-mpi pingpong-2way pingpong-2way-mpi pingpong-pipeline pingpong-pipeline-mpi *~ *.sts core ampirun

