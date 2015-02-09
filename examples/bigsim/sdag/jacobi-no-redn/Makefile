-include ../../../common.mk
OPTS=-g
CHARMC=../../../../bin/charmc $(OPTS)

all: jacobi

jacobi: parallelJacobi.o
	$(CHARMC) -language charm++ -o jacobi parallelJacobi.o -tracemode projections

parallelJacobi.o: parallelJacobi.decl.h parallelJacobi.def.h parallelJacobi.h parallelJacobi.C
	$(CHARMC) -c parallelJacobi.C 

parallelJacobi.decl.h parallelJacobi.def.h: parallelJacobi.ci
	$(CHARMC) parallelJacobi.ci

run:	jacobi
	$(call run, +p16 ./jacobi 2000 4 1000 +x10 +y10 +z10 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000 #++debug-no-pause)
#+bgwalltime 

tiny: 	jacobi
	$(call run, +p1 ./jacobi 8 4 4 +x2 +y1 +z1 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000)


small: 	jacobi
	$(call run, +p2 ./jacobi 40 4 10 +x2 +y5 +z1 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000)

medium: jacobi
	$(call run, +p5 ./jacobi 400 4 100 +x2 +y5 +z5 +cth1 +wth1 +stacksize 4000 +logsize 20000)

testnocorr:jacobi
	$(call run, +p8 ./jacobi 2000 4 1000 +x10 +y5 +z2 +cth1 +wth1 +bglog +stacksize 4000 +logsize 20000)

test: jacobi
	$(call run, +p4 ./jacobi 64 10 32 )

big: jacobi
	$(call run, +p10 ./jacobi 8000 4 4000 +x10 +y50 +z2 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 20000)

large: jacobi
	$(call run, +p15 ./jacobi 40000 4 20000 +x50 +y20 +z20 +cth1 +wth1 +bgcorrect +stacksize 4000 +logsize 200000)
	prun ./jacobi 128000 4 128000 +x40 +y40 +z20 +cth1 +wth1 +bgcorrect +stacksize 10000 +logsize 800000

bglog: jacobi
	$(call run, +p4 ./jacobi 16 10 4 +bgconfig ./bg_config)
#	./charmrun +p4 ./jacobi 16 10 4 +x2 +y2 +z2 +cth1 +wth1 +bglog

bgtest: jacobi
	$(call run, +p4 ./jacobi 64 10 32 +bgconfig ./bg_config +bgstacksize 102400 )

clean:
	rm -f *.sts *.log C *.o *.def.h *.decl.h *~ jacobi bgTraceFile
	rm -f charmrun conv-host core charmrun.exe jacobi.exe jacobi.pdb jacobi.ilk
