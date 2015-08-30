CHARMC	= ../../../../bin/charmc $(OPTS)

OBJS = jacobi3d.o

all: jacobi3d

jacobi3d: $(OBJS)
	$(CHARMC) -language charm++ -o jacobi3d $(OBJS) -tracemode autoPerf -tracemode autoTuner


projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -lz -o jacobi3d.prj $(OBJS) -tracemode autoPerf -tracemode autoTuner

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o jacobi3d.sum $(OBJS)

autoPerf: $(OBJS)
	$(CHARMC) -language charm++ -tracemode autoPerf -tracemode autoTuner -lz -o jacobi3d.ap $(OBJS)

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC)  jacobi3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi3d jacobi3d.prj charmrun tree.txt fuzzytree.txt *~

jacobi3d.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -c jacobi3d.C

test: jacobi3d
	./charmrun +p2 ./jacobi3d 20 10 +balancer Refine $(TESTOPTS)
