#CHARMDIR = /Users/yanhuasun/PPL/charm/charm-6.1.2/multicore-darwin-x86_64
OPTS = -g
CHARMC = charmc $(OPTS)  
default: all
all: sat

sat : main.o Solver.o 
	$(CHARMC) -lz -language charm++  -o sat main.o Solver.o

main.o : main.C main.decl.h main.def.h 
	$(CHARMC) -o main.o main.C 

Solver.o : Solver.C SolverTypes.h
	$(CHARMC) -o Solver.o Solver.C

#util.o : util.C
#	$(CHARMC) -o util.o util.c

main.def.h main.decl.h : main.ci
	$(CHARMC) main.ci

clean:
	rm *.o *.def.h *.decl.h
	rm  sat
