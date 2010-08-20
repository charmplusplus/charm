CHARMC = charmc $(OPTS) 
default: all
all: nqueen  
#all: nqueen  nqueen_rand nqueen_rand.prj  nqueen_neighbor nqueen_neighbor.prj nqueen_workstealing nqueen_workstealing.prj

nqueen : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -o nqueen main.o nqueen.o counter.o

nqueen_rand : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -balance rand -o nqueen_rand main.o nqueen.o counter.o

nqueen_rand.prj : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -tracemode projections -balance rand -o nqueen_rand.prj main.o nqueen.o counter.o

nqueen_neighbor : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -balance neighbor -o nqueen_neighbor main.o nqueen.o counter.o

nqueen_prj : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -tracemode projections -balance rand -o nqueen_prj main.o nqueen.o counter.o

nqueen_neighbor.prj : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -tracemode projections -balance neighbor -o nqueen_neighbor.prj main.o nqueen.o counter.o

nqueen_workstealing : main.o nqueen.o counter.o
	$(CHARMC) -language charm++  -balance workstealing -o nqueen_workstealing main.o nqueen.o counter.o

nqueen_workstealing.prj : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -tracemode projections -balance workstealing -o nqueen_workstealing.prj main.o nqueen.o counter.o

nqueen_centralized.prj : main.o nqueen.o counter.o
	$(CHARMC) -language charm++ -tracemode projections -balance centralized -o nqueen_centralized.prj main.o nqueen.o counter.o

nqueen_centralized : main.o nqueen.o counter.o
	$(CHARMC) -language charm++  -balance centralized -o nqueen_centralized main.o nqueen.o counter.o

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

test:
	 ./charmrun +p8 ./nqueen 17 5 

clean:
	rm -f main.decl.h main.def.h main.o
	rm -f counter.decl.h counter.def.h nqueen.o
	rm -f nqueen nqueen_refine nqueen_greedy nqueen_prj nqueen_rand nqueen_spray nqueen_neighbor charmrun
	rm *.o *.def.h *.decl.h
	rm *.log *.sts *.projrc

