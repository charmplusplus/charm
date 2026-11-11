-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

all: migrate_then_atsync

migrate_then_atsync: migrate_then_atsync.o
	$(CHARMC) -language charm++ -o $@ $< -module CommonLBs

migrate_then_atsync.decl.h migrate_then_atsync.def.h: migrate_then_atsync.ci
	$(CHARMC) $<

migrate_then_atsync.o: migrate_then_atsync.C migrate_then_atsync.decl.h
	$(CHARMC) -c $<

# Elements migrateMe() to the next PE, then run three AtSync steps (#4195).
test: migrate_then_atsync
	$(call run, ./migrate_then_atsync 0 +p2 +balancer RotateLB)
	$(call run, ./migrate_then_atsync 1 +p2 +balancer RotateLB)
	$(call run, ./migrate_then_atsync 2 +p2 +balancer RotateLB)
	$(call run, ./migrate_then_atsync 1 +p2 +balancer GreedyRefineLB)
	$(call run, ./migrate_then_atsync 2 +p4 +balancer GreedyRefineLB)
	$(call run, ./migrate_then_atsync 2 +p4 +balancer RotateLB)

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o migrate_then_atsync charmrun
