-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: speed

speed:  speed.o
	$(CHARMC) speed.o  -o speed -language charm++

speed.o : speed.C speed.h speed.ci
	$(CHARMC) -c speed.ci speed.C

test: speed
	$(call run, +p4 ./speed 8 )

testp: speed
	$(call run, +p$(P) ./speed $$(( $(P) * 2 )) )

clean:
	rm -f conv-host speed *.def.h *.decl.h *.o *~ charmrun charmrun.exe speed.exe speed.pdb speed.ilk
