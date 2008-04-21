CHARMDIR = /expand/home/kpattab2/charm
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)


default: all
all: molecular


molecular : main.o
  $(CHARMC) -language charm++ -o molecular main.o

main.o : main.C main.decl.h main.def.h
  $(CHARMC) -o main.o main.C

main.decl.h main.def.h : main.ci
	$(CHARMC) main.ci


clean:
  rm -f main.decl.h main.def.h main.o molecular charmrun

