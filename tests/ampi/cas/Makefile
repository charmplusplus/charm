-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: cas

cas: test.o
	$(CHARMC) -o cas test.o

test.o: test.C
	$(CHARMC) -c test.C

clean:
	rm -f *.o cas ampirun charmrun conv-host *~

test: cas
	$(call run, ./cas +p1 +vp1)
	$(call run, ./cas +p1 +vp2)
	$(call run, ./cas +p1 +vp4)
	$(call run, ./cas +p2 +vp2)
	$(call run, ./cas +p2 +vp4)

testp: cas
	$(call run, ./cas +p$(P) +vp$(P) )
	$(call run, ./cas +p$(P) +vp$$(( $(P) * 2 )) )
