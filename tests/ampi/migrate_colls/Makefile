-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: migrate_colls

migrate_colls: test.o
	$(CHARMC) -o migrate_colls test.o

test.o: test.C
	$(CHARMC) -c test.C

# Default migration mode (no +ampiMigrateWithCollectives). Collectives whose
# results travel as messages to the rank, and every collective on a
# communicator other than MPI_COMM_WORLD, may be outstanding across
# AMPI_Migrate; RotateLB moves every rank, so the results have to follow them.
# A collective whose result is a broadcast to MPI_COMM_WORLD's array must make
# AMPI_Migrate abort, naming the option.
ABORT_MSG = "non-blocking collective outstanding"
test: migrate_colls
	$(call run, ./migrate_colls +p1 +vp4 )
	$(call run, ./migrate_colls +p1 +vp4 +balancer RotateLB )
	$(call run, ./migrate_colls +p2 +vp4 +balancer RotateLB )
	$(call run, ./migrate_colls +p2 +vp7 +balancer RotateLB )
	$(call run, ./migrate_colls +p1 +vp4 abort iallreduce ) 2>&1 | grep -q $(ABORT_MSG)
	$(call run, ./migrate_colls +p1 +vp4 abort iallgather ) 2>&1 | grep -q $(ABORT_MSG)
	$(call run, ./migrate_colls +p1 +vp4 abort iallgatherv ) 2>&1 | grep -q $(ABORT_MSG)
	$(call run, ./migrate_colls +p1 +vp4 abort ibarrier ) 2>&1 | grep -q $(ABORT_MSG)
	$(call run, ./migrate_colls +p1 +vp4 abort ibcast ) 2>&1 | grep -q $(ABORT_MSG)

testp: migrate_colls
	$(call run, ./migrate_colls +p$(P) +vp$(P) +balancer RotateLB )
	$(call run, ./migrate_colls +p$(P) +vp$$(( $(P) * 2 )) +balancer RotateLB )

clean:
	rm -f *.o *.mod migrate_colls *~ conv-host charmrun ampirun migrate_colls.exe migrate_colls.pdb migrate_colls.ilk
