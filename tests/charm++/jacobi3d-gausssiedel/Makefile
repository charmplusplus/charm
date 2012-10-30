CHARMC=../../../bin/charmc $(OPTS) $(MOPTS) -O3 -g

OBJS = main.o 
OBJS_J = main.o

OBJS_OMP = main_omp.o

OMP_FLAG = -DJACOBI=1 -DJACOBI_OPENMP -fopenmp

all: main main.j main.ckloop

projections: main.ckloop.prj main.j.prj

main: $(OBJS)
	$(CHARMC) -language charm++ -module DummyLB -o main $(OBJS)

main.j: $(OBJS_J)
	$(CHARMC) -language charm++ -module DummyLB -o main.j $(OBJS_J)

main.j.prj: $(OBJS_J)
	$(CHARMC) -language charm++ -module DummyLB -tracemode projections -o main.j.prj $(OBJS_J)

main.ckloop: main_ckloop.o
	$(CHARMC) -language charm++ -module DummyLB -module CkLoop -o main.ckloop main_ckloop.o

main.ckloop.prj: main_ckloop.o
	$(CHARMC) -language charm++ -tracemode projections  -module DummyLB -module CkLoop -o main.ckloop.prj main_ckloop.o

main_omp: $(OBJS_OMP)
	$(CHARMC) -language charm++ -module DummyLB $(OMP_FLAG) -o main_omp $(OBJS_OMP)

projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -lz -o main.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o main.sum $(OBJS)

main.decl.h: main.ci
	$(CHARMC)  -E main.ci

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC)   jacobi3d.ci

syncfttest causalfttest: fttest
fttest: main
	./charmrun ./main 256 128 +p8 +balancer DummyLB $(TESTOPTS)
	./charmrun ./main 256 128 +p8 +balancer DummyLB +killFile kill_01.txt $(TESTOPTS)
	./charmrun ./main 256 256 256 64 64 32 +p8 +balancer DummyLB +killFile kill_02.txt $(TESTOPTS)
	./charmrun ./main 256 256 256 64 64 32 +p8 +balancer DummyLB +killFile kill_03.txt $(TESTOPTS)


clean:
	rm -f *.decl.h *.def.h conv-host *.o $(all) main main.prj charmrun *~

main.o: main.cc jacobi3d.decl.h main.decl.h
	$(CHARMC) -c main.cc

main_j.o: main.cc main.decl.h  jacobi3d.decl.h
	$(CHARMC) -DJACOBI=1 -c main.cc -o main_j.o

main_ckloop.o: main.cc main.decl.h  jacobi3d.decl.h
	$(CHARMC) -DJACOBI=1 -DCKLOOP=1 -c main.cc -o main_ckloop.o

main_omp.o: main.cc main.decl.h  jacobi3d.decl.h
	$(CHARMC) $(OMP_FLAG) -o $(OBJS_OMP) -c main.cc

