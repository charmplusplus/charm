-include ../../common.mk
CHARMC=../../../bin/ampicc $(OPTS)

all: stacksize

stacksize: test.o
	$(CHARMC) -o stacksize test.o

test.o: test.c
	$(CHARMC) -c test.c

clean:
	rm -f *.o *.mod stacksize *~ conv-host charmrun charmrun.exe stacksize.exe stacksize.pdb stacksize.ilk ampirun

test: stacksize
	$(call run, ./stacksize +p2 +vp4 )

testp: stacksize
	$(call run, ./stacksize +p$(P) +vp$(P) )
	$(call run, ./stacksize +p$(P) +vp$$(( $(P) * 2 )) )
	$(call run, ./stacksize +p$(P) +vp$$(( $(P) * 4 )) )
