-include ../../common.mk
LINKOPTS=-memory os
CHARMC=../../../bin/charmc $(OPTS)

OBJS = kNeighbor.o
COMPOBJS = kNeighbor.comp.o

all: kNeighbor 

projections: kNeighbor.proj kNeighbor.comp.proj

kNeighbor: $(OBJS)
	$(CHARMC) -language charm++ -o kNeighbor $(OBJS) 
	$(CHARMC) -language charm++ -o kNeighbor.memos $(OBJS) $(LINKOPTS)

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

test: all
	$(call run, +p5 ./kNeighbor 5 10 64 )
	$(call run, +p5 ./kNeighbor 5 10 256 )
	$(call run, +p5 ./kNeighbor 5 10 1024 )
	$(call run, +p5 ./kNeighbor 5 10 16384 )

test-smp: all
	$(call run, +p5 ./kNeighbor.memos +setcpuaffinity 5 10000 64 ++ppn 5)
	$(call run, +p5 ./kNeighbor.memos +setcpuaffinity 5 10000 256 ++ppn 5)
	$(call run, +p5 ./kNeighbor.memos +setcpuaffinity 5 10000 1024 ++ppn 5)
	$(call run, +p5 ./kNeighbor.memos +setcpuaffinity 5 10000 16384 ++ppn 5)

test-pxshm: all
	$(call run, +p5 ./kNeighbor +setcpuaffinity 5 10000 64)
	$(call run, +p5 ./kNeighbor +setcpuaffinity 5 10000 256)
	$(call run, +p5 ./kNeighbor +setcpuaffinity 5 10000 1024)
	$(call run, +p5 ./kNeighbor +setcpuaffinity 5 10000 16384)
