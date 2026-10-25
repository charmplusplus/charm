-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: collmatch

collmatch: collmatch.o
	$(CHARMC) -o collmatch collmatch.o

collmatch.o: collmatch.C
	$(CHARMC) -c collmatch.C

# Part 8 of the test keeps a non-blocking collective outstanding across
# AMPI_Migrate. Once AMPI's two migration modes land (charm #4072 item 8),
# that is allowed only with +ampiMigrateWithCollectives; before they land
# the flag is merely reported as unparsed. So every run carries it.
test: collmatch
	$(call run, ./collmatch +p1 +vp4 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p1 +vp7 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p2 +vp4 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p2 +vp7 +ampiMigrateWithCollectives)

# Not in `make test` until the migration-with-reductions-in-flight problem
# recorded on charm #4072 is fixed in the core; run by hand.
test-lb: collmatch
	$(call run, ./collmatch +p2 +vp4 +balancer RandCentLB +ampiMigrateWithCollectives)

testp: collmatch
	$(call run, ./collmatch +p$(P) +vp$(P) )
	$(call run, ./collmatch +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod collmatch *~ conv-host charmrun ampirun collmatch.exe collmatch.pdb collmatch.ilk
