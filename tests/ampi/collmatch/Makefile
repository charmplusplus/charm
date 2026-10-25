-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: collmatch

collmatch: collmatch.o
	$(CHARMC) -o collmatch collmatch.o

collmatch.o: collmatch.C
	$(CHARMC) -c collmatch.C

test: collmatch
	$(call run, ./collmatch +p1 +vp4)
	$(call run, ./collmatch +p1 +vp7)
	$(call run, ./collmatch +p2 +vp4)
	$(call run, ./collmatch +p2 +vp7)

# Not in `make test` until the migration-with-reductions-in-flight problem
# recorded on charm #4072 is fixed in the core; run by hand.
test-lb: collmatch
	$(call run, ./collmatch +p2 +vp4 +balancer RandCentLB)

testp: collmatch
	$(call run, ./collmatch +p$(P) +vp$(P) )
	$(call run, ./collmatch +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod collmatch *~ conv-host charmrun ampirun collmatch.exe collmatch.pdb collmatch.ilk
