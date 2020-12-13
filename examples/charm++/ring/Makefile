-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: ring

ring:  ring.o
	$(CHARMC) ring.o  -o ring -language charm++

ring.o : ring.C ring.h ring.ci
	$(CHARMC) -c ring.ci ring.C

test: ring
	$(call run, +p4 ./ring )

testp: ring
	$(call run, +p$(P) ./ring )

clean:
	rm -f conv-host ring *.def.h *.decl.h *.o *~ charmrun charmrun.exe ring.exe ring.pdb ring.ilk
