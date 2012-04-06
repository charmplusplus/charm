CHARMC=../../../bin/charmc $(OPTS) $(MOPTS)

OBJS = jacobi3d.o

all: jacobi3d

jacobi3d: $(OBJS)
	$(CHARMC) -language charm++ -module DummyLB -o jacobi3d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -lz -o jacobi3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o jacobi3d.sum $(OBJS)

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC)  jacobi3d.ci

fttest: jacobi3d
	./charmrun ./jacobi3d 256 128 +p8 +balancer DummyLB
	./charmrun ./jacobi3d 256 128 +p8 +balancer DummyLB +killFile kill_01.txt
	./charmrun ./jacobi3d 256 256 256 64 64 32 +p8 +balancer DummyLB +killFile kill_02.txt
	./charmrun ./jacobi3d 256 256 256 64 64 32 +p8 +balancer DummyLB +killFile kill_03.txt

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi3d jacobi3d.prj charmrun *~

jacobi3d.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -c jacobi3d.C
