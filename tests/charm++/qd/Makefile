-include ../../common.mk
CHARMC := ../../../bin/charmc

all: qdbench qdstress

qdbench: qdbench.o
	$(CHARMC) $(OPTS) -o $@ qdbench.o

qdbench.o: qdbench.C qdbench.decl.h
	$(CHARMC) $(OPTS) -c qdbench.C

qdbench.decl.h: qdbench.ci
	$(CHARMC) qdbench.ci

# +balancer RotateLB needs the CommonLBs module; the CompletionDetector library
# is the completion module.
qdstress: qdstress.o
	$(CHARMC) $(OPTS) -language charm++ -o $@ qdstress.o -module completion -module CommonLBs

qdstress.o: qdstress.C qdstress.decl.h
	$(CHARMC) $(OPTS) -c qdstress.C

qdstress.decl.h: qdstress.ci
	$(CHARMC) qdstress.ci

# qdstress runs every phase (see the top of qdstress.C). The +qd 1 line (fake
# QD: the callback fires 1 s after CkStartQD reaches PE 0) runs only the
# fakeqd phase, which checks that the callback fires and when; it runs in one
# process, as the second pass would repeat it.
test: all
	$(call run, ./qdbench +p2)
	$(call run, ./qdbench +p4)
	$(call run, ./qdstress +p1 )
	$(call run, ./qdstress +p4 +balancer RotateLB -rotate )
	$(call run, ./qdstress +p2 +randomized_msgq +randomized_seed 1 )
	$(call run, ./qdstress +p4 +randomized_msgq +randomized_seed 2 )
	$(call run, ./qdstress +p4 +balancer RotateLB -rotate +randomized_msgq +randomized_seed 3 )
	TESTRUN_PROCS=1 $(call run, ./qdstress +p2 +qd 1 -rounds 2 )

testp: all
	$(call run, ./qdbench +p$(P))
	$(call run, ./qdstress +p$(P) +balancer RotateLB -rotate )

smptest: all
	$(call run, ./qdbench +p4 ++ppn 2)
	$(call run, ./qdstress +p4 ++ppn 2 )

clean:
	rm -f *.decl.h *.def.h *.o qdbench qdstress charmrun
