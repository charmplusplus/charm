-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: bindasync

bindasync: bindasync.decl.h bindasync.C
	$(CHARMC) bindasync.C -o bindasync

bindasync.decl.h: bindasync.ci
	$(CHARMC) bindasync.ci

# B's creation request can reach PE 0 before or after the map's creation; the
# randomized queue produces both orders over the seeds.
test: bindasync
	$(call run, ./bindasync +p4)
	$(call run, ./bindasync +p4 +randomized_msgq +randomized_seed 1)
	$(call run, ./bindasync +p4 +randomized_msgq +randomized_seed 2)
	$(call run, ./bindasync +p4 +randomized_msgq +randomized_seed 3)
	$(call run, ./bindasync +p4 +randomized_msgq +randomized_seed 4)
	$(call run, ./bindasync +p2 +randomized_msgq +randomized_seed 5)

smptest: bindasync
	$(call run, ./bindasync +p4 +randomized_msgq +randomized_seed 1)

clean:
	rm -f *.decl.h *.def.h *.o bindasync charmrun
