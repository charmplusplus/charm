-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = overlapTest.o overlapTestCU.o

NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -O3 -c -use_fast_math #-device-debug -deviceemu 
NVCC_INC = -I/usr/local/cuda/include  -I../../../../../src/arch/cuda/hybridAPI
CHARMINC = -I../../../../include
LD_LIBS= -lcublas
all: overlapTest

overlapTest: $(OBJS)
	$(CHARMC) -language charm++ -o overlapTest $(OBJS) $(LD_LIBS)

overlapTest.decl.h: overlapTest.ci
	$(CHARMC)  overlapTest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o overlapTest charmrun

overlapTest.o: overlapTest.C overlapTest.decl.h
	$(CHARMC) -O3 -c overlapTest.C

overlapTestCU.o: overlapTest.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARMINC) -o overlapTestCU.o overlapTest.cu

test: all
	$(call run, ./overlapTest +p2 2 8)
