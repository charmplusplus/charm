OPTS	= -O3
CHARMC	= ../../../bin/charmc $(OPTS)

OBJS = jacobi2d.o

all: jacobi2d

jacobi2d: $(OBJS)
	$(CHARMC) -language charm++ -module RefineLB -o jacobi2d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -lz -o jacobi2d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o jacobi2d.sum $(OBJS)

jacobi2d.decl.h: jacobi2d.ci
	$(CHARMC)  jacobi2d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi2d jacobi2d.prj charmrun *~

jacobi2d.o: jacobi2d.C jacobi2d.decl.h
	$(CHARMC) -c jacobi2d.C

test: jacobi2d
	./charmrun +p2 ./jacobi2d 20 10 +balancer Refine $(TESTOPTS)
