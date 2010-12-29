# Makefile for load-banacing test program

CHARMC= ../../../../bin/charmc $(OPTS)

OBJS = sim.o Topo.o

all:	cifiles sim 

sim: $(OBJS)
	$(CHARMC) -language charm++ -o sim $(OBJS)

cifiles: sim.ci Topo.ci
	$(CHARMC)  sim.ci Topo.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o sim charmrun

sim.o: sim.C sim.decl.h
	$(CHARMC) -c -g sim.C 

Topo.o: Topo.C Topo.decl.h Topo.h
	$(CHARMC) -c -g Topo.C
