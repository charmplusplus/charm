OPTS=-O3
#OPTS=-O3 -DRAW_STENCIL
CHARMC=charmc $(OPTS)

all: jacobi

jacobi: parallelJacobi.o
	$(CHARMC) -language charm++ -o jacobi parallelJacobi.o

parallelJacobi.o: parallelJacobi.decl.h parallelJacobi.def.h parallelJacobi.h parallelJacobi.C
	$(CHARMC) -c parallelJacobi.C 

parallelJacobi.decl.h parallelJacobi.def.h: parallelJacobi.ci
	$(CHARMC) parallelJacobi.ci

run:	jacobi
	./charmrun +p16 ./jacobi 2000 4 1000 +x10 +y10 +z10 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000 #++debug-no-pause
#+bgwalltime 

tiny: 	jacobi
	./charmrun +p1 ./jacobi 8 4 4 +x2 +y1 +z1 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000


small: 	jacobi
	./charmrun +p2 ./jacobi 40 4 10 +x2 +y5 +z1 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000

medium: jacobi
	./charmrun +p5 ./jacobi 400 4 100 +x2 +y5 +z5 +cth1 +wth1 +stacksize 4000 +logsize 20000

testnocorr:jacobi
	./charmrun +p8 ./jacobi 2000 4 1000 +x10 +y5 +z2 +cth1 +wth1 +bglog +stacksize 4000 +logsize 20000

test: jacobi
	./charmrun +p4 ./jacobi 64 10 32 $(TESTOPTS)

big: jacobi
	./charmrun +p10 ./jacobi 8000 4 4000 +x10 +y50 +z2 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000

large: jacobi
	 ./charmrun +p15 ./jacobi 40000 4 20000 +x50 +y20 +z20 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 200000
	prun ./jacobi 128000 4 128000 +x40 +y40 +z20 +cth1 +wth1 +bgcorrect +stacksize 10000 +logsize 800000

clean:
	rm -f *.sts *.log C *.o *.def.h *.decl.h *~ jacobi bgTraceFile
	rm -f charmrun conv-host core charmrun.exe jacobi.exe jacobi.pdb jacobi.ilk
