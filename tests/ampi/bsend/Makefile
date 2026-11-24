-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: bsend

bsend: bsend.o
	$(CHARMC) -o bsend bsend.o

bsend.o: bsend.C
	$(CHARMC) -c bsend.C

test: bsend
	$(call run, ./bsend +p1 +vp2)
	$(call run, ./bsend +p2 +vp2)

testp: bsend
	$(call run, ./bsend +p$(P) +vp$(P) )

clean:
	rm -f *.o *.mod bsend *~ conv-host charmrun bsend.o charmrun.exe bsend.exe bsend.pdb bsend.ilk ampirun
