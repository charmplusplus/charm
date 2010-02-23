CHARMDIR = /expand/home/jessie/PPL/git/charm/mpi-linux-x86_64
#CHARMDIR = /Users/yanhuasun/PPL/charm/charm-6.1.2/multicore-darwin-x86_64
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS) 
default: all
all: nqueen nqueen_prj nqueen_greedy nqueen_refine nqueen_spray nqueen_neighbor

nqueen : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -o nqueen main.o nqueen.o counter.o

nqueen_spray : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -balance spray -o nqueen_spray main.o nqueen.o counter.o

nqueen_neighbor : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -balance neighbor -o nqueen_neighbor main.o nqueen.o counter.o

nqueen_prj : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -tracemode projections -o nqueen_prj main.o nqueen.o counter.o

nqueen_greedy : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -balancer GreedyCommLB  -o nqueen_greedy main.o nqueen.o counter.o

nqueen_refine : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -balancer RefineLB  -o nqueen_refine main.o nqueen.o counter.o
main.o : main.C main.h counter.decl.h counter.def.h  main.decl.h main.def.h
	$(CHARMC) -o main.o main.C

nqueen.o : nqueen.C nqueen.h counter.decl.h counter.def.h main.decl.h
	$(CHARMC) -o nqueen.o nqueen.C

counter.o: counter.C
	$(CHARMC) -o counter.o counter.C

main.decl.h main.def.h : main.ci
	$(CHARMC) main.ci
	
counter.decl.h counter.def.h : counter.ci
	$(CHARMC) counter.ci

clean:
	rm -f main.decl.h main.def.h main.o
	rm -f counter.decl.h counter.def.h nqueen.o
	rm -f nqueen nqueen_refine nqueen_greedy nqueen_prj nqueen_rand nqueen_spray nqueen_neighbor charmrun
	rm *.o *.def.h *.decl.h
	rm *.log *.sts *.projrc

