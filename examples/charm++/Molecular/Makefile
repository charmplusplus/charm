CHARMDIR = /expand/home/kpattab2/charm
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)


default: all
all: molecular


molecular : main.o
	$(CHARMC)	-language	charm++	-g -o molecular main.o -module liveViz

main.o : main.C	main.decl.h	main.def.h
	$(CHARMC) -g -o main.o main.C

main.decl.h main.def.h : main.ci
	$(CHARMC) main.ci

clean:
	rm -f main.decl.h main.def.h main.o hello charmrun
