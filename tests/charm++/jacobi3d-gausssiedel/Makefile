CHARMC=../../../bin/charmc $(OPTS) $(MOPTS) -O3 -g

OBJS = jacobi3d.o 
OBJS_J = jacobi3d_j.o

OBJS_OMP = jacobi3d_omp.o

OMP_FLAG = -DJACOBI=1 -DJACOBI_OPENMP -fopenmp

all: jacobi3d jacobi3d.j jacobi3d.ckloop

projections: jacobi3d.ckloop.prj jacobi3d.j.prj

jacobi3d: $(OBJS)
	$(CHARMC) -language charm++ -module DummyLB -o jacobi3d $(OBJS)

jacobi3d.j: $(OBJS_J)
	$(CHARMC) -language charm++ -module DummyLB -o jacobi3d.j $(OBJS_J)

jacobi3d.j.prj: $(OBJS_J)
	$(CHARMC) -language charm++ -module DummyLB -tracemode projections -o jacobi3d.j.prj $(OBJS_J)

jacobi3d.ckloop: jacobi3d_ckloop.o
	$(CHARMC) -language charm++ -module DummyLB -module CkLoop -o jacobi3d.ckloop jacobi3d_ckloop.o

jacobi3d.ckloop.prj: jacobi3d_ckloop.o
	$(CHARMC) -language charm++ -tracemode projections  -module DummyLB -module CkLoop -o jacobi3d.ckloop.prj jacobi3d_ckloop.o

jacobi3d_omp: $(OBJS_OMP)
	$(CHARMC) -language charm++ -module DummyLB $(OMP_FLAG) -o jacobi3d_omp $(OBJS_OMP)

projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -lz -o jacobi3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o jacobi3d.sum $(OBJS)

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC)  -E jacobi3d.ci

syncfttest causalfttest: fttest
fttest: jacobi3d
	./charmrun ./jacobi3d 256 128 +p8 +balancer DummyLB $(TESTOPTS)
	./charmrun ./jacobi3d 256 128 +p8 +balancer DummyLB +killFile kill_01.txt $(TESTOPTS)
	./charmrun ./jacobi3d 256 256 256 64 64 32 +p8 +balancer DummyLB +killFile kill_02.txt $(TESTOPTS)
	./charmrun ./jacobi3d 256 256 256 64 64 32 +p8 +balancer DummyLB +killFile kill_03.txt $(TESTOPTS)


clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi3d jacobi3d.prj charmrun *~

jacobi3d.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -c jacobi3d.C

jacobi3d_j.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -DJACOBI=1 -c jacobi3d.C -o jacobi3d_j.o

jacobi3d_ckloop.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -DJACOBI=1 -DCKLOOP=1 -c jacobi3d.C -o jacobi3d_ckloop.o

jacobi3d_omp.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) $(OMP_FLAG) -o $(OBJS_OMP) -c jacobi3d.C

