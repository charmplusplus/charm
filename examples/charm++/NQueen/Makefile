-include ../../common.mk
CHARMC = ../../../bin/charmc -O3 $(OPTS) 
default: all
all: nqueen nqueen_neighbor nqueen_workstealing

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

main.o : main.C main.h Counter.decl.h main.decl.h main.def.h
	$(CHARMC) -o main.o main.C

nqueen.o : nqueen.C nqueen.h Counter.decl.h main.decl.h
	$(CHARMC) -o nqueen.o nqueen.C

counter.o: counter.C  Counter.decl.h Counter.def.h
	$(CHARMC) -o counter.o counter.C

%.stamp: %
	$(CHARMC) $<
	touch $@

main.decl.h main.def.h : main.ci.stamp
Counter.decl.h Counter.def.h : counter.ci.stamp

test: nqueen
	$(call run, +p 4 ./nqueen 16 4 )

bgtest: all
	$(call run, +p 4 ./nqueen 16 4 +x2 +y2 +z2 +bgstacksize 102400 )

clean:
	rm -f main.decl.h main.def.h main.o
	rm -f Counter.decl.h Counter.def.h nqueen.o
	rm -f nqueen nqueen_refine nqueen_greedy nqueen_prj nqueen_rand nqueen_spray nqueen_neighbor charmrun nqueen_*
	rm -f *.o *.def.h *.decl.h *.stamp
	rm -f *.log *.sts *.projrc

