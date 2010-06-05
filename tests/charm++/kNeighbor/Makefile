LINKOPTS=-memory os
CHARMC=../../../bin/charmc $(OPTS)

OBJS = kNeighbor.o 
COMPOBJS = kNeighbor.comp.o

all: kNeighbor 

projections: kNeighbor.proj kNeighbor.comp.proj

kNeighbor: $(OBJS)
	$(CHARMC) -language charm++ -o kNeighbor $(OBJS)  -lbigsim-param

traceBigSim.o:  traceBigSim.C
	$(CHARMC) -c traceBigSim.C

kNeighbor.comp: $(COMPOBJS)
	$(CHARMC) -language charm++ -o kNeighbor.comp.memos $(COMPOBJS) 
	$(CHARMC) -language charm++ -o kNeighbor.comp $(COMPOBJS) $(LINKOPTS)

kNeighbor.proj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o kNeighbor.proj $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o kNeighbor.memos.proj $(OBJS) $(LINKOPTS)

kNeighbor.comp.proj: $(COMPOBJS)
	$(CHARMC) -language charm++ -tracemode projections -o kNeighbor.comp.proj $(COMPOBJS)

kNeighbor.decl.h: kNeighbor.ci
	$(CHARMC)  kNeighbor.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o charmrun *~ kNeighbor kNeighbor.memos kNeighbor.comp kNeighbor.comp.memos kNeighbor.proj kNeighbor.comp.proj

kNeighbor.o: kNeighbor.C kNeighbor.decl.h 
	$(CHARMC) -c kNeighbor.C

kNeighbor.comp.o: kNeighbor.C kNeighbor.decl.h
	$(CHARMC) -c -DDOCOMP kNeighbor.C -o kNeighbor.comp.o


test-smp: all
	./charmrun +p7 ./kNeighbor.memos +setcpuaffinity 7 10000 64 ++ppn 7
	./charmrun +p7 ./kNeighbor.memos +setcpuaffinity 7 10000 256 ++ppn 7
	./charmrun +p7 ./kNeighbor.memos +setcpuaffinity 7 10000 1024 ++ppn 7
	./charmrun +p7 ./kNeighbor.memos +setcpuaffinity 7 10000 16384 ++ppn 7

test-pxshm: all
	./charmrun +p7 ./kNeighbor +setcpuaffinity 7 10000 64 +nodesize 7
	./charmrun +p7 ./kNeighbor +setcpuaffinity 7 10000 256 +nodesize 7
	./charmrun +p7 ./kNeighbor +setcpuaffinity 7 10000 1024 +nodesize 7
	./charmrun +p7 ./kNeighbor +setcpuaffinity 7 10000 16384 +nodesize 7


interpolate:
	./charmrun +p3 ./kNeighbor 3 1 64 +x3 +y1 +z1 +bglog
	./charmrun +p3 ./kNeighbor 3 1 64

