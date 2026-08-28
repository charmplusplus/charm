-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: bcastred

bcastred: bcastred.decl.h bcastred.C
	$(CHARMC) bcastred.C -o bcastred

bcastred.decl.h: bcastred.ci
	$(CHARMC) bcastred.ci

test: bcastred
	$(call run, +p2 ./bcastred)
	$(call run, +p2 ./bcastred -n 8 -i 200 -m 1 -c 8 )

# Scaling run. The ring is sized by -n, independently of the PE count, so any
# P works; the migration targets are what actually vary with P.
testp: bcastred
	$(call run, +p$(P) ./bcastred)
	$(call run, +p$(P) ./bcastred -n 8 -i 200 -m 1 -c 8 )

# The interesting case for an anytime-migration test: with more than one PE
# per process, elements migrate within a process as well as between them.
# +p2 ++ppn 2 is one process, +p4 ++ppn 2 is two.
smptest: bcastred
	$(call run, +p2 ++ppn 2 ./bcastred)
	$(call run, +p4 ++ppn 2 ./bcastred -n 8 -i 200 -m 1 -c 8 )

clean:
	rm -rf *.decl.h *.def.h bcastred charmrun
