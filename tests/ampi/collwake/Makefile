-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: collwake

collwake: collwake.o
	$(CHARMC) -o collwake collwake.o

collwake.o: collwake.C
	$(CHARMC) -c collwake.C

test: collwake
	$(call run, ./collwake +p1 +vp4)
	$(call run, ./collwake +p2 +vp4)

testp: collwake
	$(call run, ./collwake +p$(P) +vp$(P) )
	$(call run, ./collwake +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod collwake *~ conv-host charmrun ampirun collwake.exe collwake.pdb collwake.ilk
