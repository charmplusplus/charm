CHARMDIR = ../../../..
OPTS = -g -Wno-deprecated
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: pgm

pgm: Main.o sectionReduction.o
	$(CHARMC) -memory charmdebug -o pgm sectionReduction.o Main.o -module CkMulticast 

Main.o: Main.C Main.h main.decl.h main.def.h sectionReduction.decl.h
	$(CHARMC) -c Main.C

sectionReduction.o: sectionReduction.C sectionReduction.h sectionReduction.decl.h sectionReduction.def.h main.decl.h
	$(CHARMC) -c sectionReduction.C

main.decl.h main.def.h: main.ci
	$(CHARMC) main.ci

sectionReduction.decl.h sectionReduction.def.h: sectionReduction.ci
	$(CHARMC) sectionReduction.ci

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f pgm charmrun
