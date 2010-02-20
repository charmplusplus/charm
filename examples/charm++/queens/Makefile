CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm:   counter.o pgm.o
	$(CHARMC) counter.o pgm.o -o pgm -language charm++ 

pgm.o : pgm.C pgm.h Pgm.def.h Pgm.decl.h
	$(CHARMC) -c pgm.C

counter.o : counter.C counter.h Counter.def.h Counter.decl.h
	$(CHARMC) -c counter.C

Pgm.def.h Pgm.decl.h : pgm.ci
	$(CHARMC) pgm.ci

Counter.def.h Counter.decl.h : counter.ci
	$(CHARMC) counter.ci

clean:
	rm -f conv-host *.o charmrun
	rm -f *.def.h *.decl.h
	rm -f pgm pgm.*.log pgm.sts charmrun.exe pgm.exe pgm.pdb pgm.ilk
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers


test: pgm
	./charmrun ./pgm +p4 10 7 $(TESTOPTS)

bgtest: pgm
	./charmrun ./pgm +p4 10 7 +x2 +y2 +z2 +bgstacksize 102400 $(TESTOPTS)

