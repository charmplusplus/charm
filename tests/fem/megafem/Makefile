-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) 
OBJS=pgm.o ctests.o 
FOBJS=$(OBJS) ftests.o
COBJS=$(OBJS) ftests_fallback.o

all: pgm 

pgm: $(COBJS)
	$(CHARMC) -o pgm $(COBJS) -language ampi -module fem

fpgm: $(FOBJS)
	$(CHARMC) -o pgm $(FOBJS) -language ampif -module fem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

ctests.o: ctests.C *.tst
	$(CHARMC) -c ctests.C

ftests.o: ftests.F90 *.tst
	$(CHARMC) -c ftests.F90 -o $@

ftests_fallback.o: ftests_fallback.C
	$(CHARMC) -c ftests_fallback.C -o $@

clean:
	rm -f pgm fpgm *.o *.MOD *.mod charmrun

test: pgm
	$(call run, ./pgm +p1 +vp1)
	$(call run, ./pgm +p1 +vp2)
	$(call run, ./pgm +p2 +vp1)
	$(call run, ./pgm +p2 +vp2)
	$(call run, ./pgm +p2 +vp3)
	$(call run, ./pgm +p2 +vp4)
	-@rm -f fem_mesh_vp3_*
	$(call run, ./pgm +p1 +vp3 -write)
	$(call run, ./pgm +p1 +vp3 -read)
	rm -f fem_mesh_vp3_*

test-mig:
	rm -fr pgm pgm.o
	make OPTS="-DENABLE_MIG=1 -memory isomalloc -balancer RandCentLB" pgm
	$(call run, ./pgm +p2)
