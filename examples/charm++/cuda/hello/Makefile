CHARMC=../../../../bin/charmc $(OPTS)

OBJS = hello.o helloCUDA.o

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -c -use_fast_math #-device-debug -deviceemu 
NVCC_INC = -I/usr/local/cuda/include -I../../../../../NVIDIA_CUDA_SDK/common/inc

LDLIBS += -lcuda -lcudart -lGL -lGLU -lcutil

export LD_RUN_PATH = /usr/local/cuda/lib

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS) $(LD_LIBS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o wr.h hello charmrun

hello.o: hello.C hello.decl.h
	$(CHARMC) -g -c hello.C

helloCUDA.o: helloCUDA.cu
	cp ../../../../src/arch/cuda/hybridAPI/wr.h .
	$(NVCC) -g $(NVCC_FLAGS) $(NVCC_INC) helloCUDA.cu

test: all
	./charmrun hello +p4 10

bgtest: all
	./charmrun hello +p4 10 +x2 +y2 +z1
