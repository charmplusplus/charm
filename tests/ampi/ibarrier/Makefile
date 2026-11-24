-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: ibarrier

ibarrier: test.o
	$(CHARMC) -o ibarrier test.o

test.o: test.C
	$(CHARMC) -c test.C

clean:
	rm -f *.o ibarrier ampirun charmrun conv-host *~

test: ibarrier
	$(call run, ./ibarrier +p1 +vp1)
	$(call run, ./ibarrier +p1 +vp2)
	$(call run, ./ibarrier +p1 +vp4)
	$(call run, ./ibarrier +p2 +vp2)
	$(call run, ./ibarrier +p2 +vp4)

testp: ibarrier
	$(call run, ./ibarrier +p$(P) +vp$(P) )
	$(call run, ./ibarrier +p$(P) +vp$$(( $(P) * 2 )) )
