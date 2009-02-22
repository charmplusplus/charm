OPTS	= -O2
CHARMC	= ../../../bin/charmc -module liveViz $(OPTS)

default: all
all: molecular

molecular : main.o cell.o
	$(CHARMC) -language charm++ -module liveViz -o molecular main.o cell.o

main.o : main.C	main.h main.decl.h main.def.h cell.decl.h common.h
	$(CHARMC) -o main.o main.C

main.decl.h main.def.h : main.ci
	$(CHARMC) main.ci

main.h : cell.decl.h
	
cell.o : cell.C cell.h cell.decl.h cell.def.h main.decl.h common.h
	$(CHARMC) -o cell.o cell.C

cell.decl.h cell.def.h : cell.ci
	$(CHARMC) cell.ci

clean:
	rm -f main.decl.h main.def.h main.o hello charmrun
	rm -f cell.decl.h cell.def.h cell.o
	rm -f molecular charmrun
