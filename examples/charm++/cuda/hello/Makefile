-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = hello.o helloCUDA.o

NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -c -use_fast_math #-device-debug -deviceemu 
NVCC_INC = -I/usr/local/cuda/include 
CHARMINC = -I../../../../include
LD_LIBS += -lcuda -lcudart 

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS) $(LD_LIBS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

helloCUDA.o: helloCUDA.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARMINC) helloCUDA.cu

test: all
	$(call run, ./hello +p4 10)

bgtest: all
	$(call run, ./hello +p4 10 +x2 +y2 +z1)
