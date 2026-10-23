-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: scan

scan: scan.o
	$(CHARMC) -o scan scan.o

scan.o: scan.C
	$(CHARMC) -c scan.C

test: scan
	$(call run, ./scan +p1 +vp4)
	$(call run, ./scan +p1 +vp3)
	$(call run, ./scan +p1 +vp5)
	$(call run, ./scan +p1 +vp7)
	$(call run, ./scan +p2 +vp4)

testp: scan
	$(call run, ./scan +p$(P) +vp$(P) )
	$(call run, ./scan +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod scan *~ conv-host charmrun ampirun scan.exe scan.pdb scan.ilk
