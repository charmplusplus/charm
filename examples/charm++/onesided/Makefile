-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm:  onesided.o
	$(CHARMC) onesided.o  -o onesidedtest -language charm++

onesided.o : onesided.C onesided.h onesided.ci
	$(CHARMC) -c onesided.ci 
	$(CHARMC) -c onesided.C

test: pgm
	$(call run, ./onesidedtest 0 0 +p2)
	$(call run, ./onesidedtest 1 0 +p2)
	$(call run, ./onesidedtest 0 1 +p2)
	$(call run, ./onesidedtest 1 1 +p2)

clean:
	rm -f conv-host onesidedtest *.def.h *.decl.h *.o *~ charmrun
