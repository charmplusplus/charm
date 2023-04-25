-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: amr_1d_random

amr_1d_random: random_amr.decl.h amr_1d_random.C
	$(CHARMC) amr_1d_random.C -o amr_1d_random 

random_amr.decl.h: amr_1d_random.ci
	$(CHARMC) amr_1d_random.ci

test: amr_1d_random
	$(call run, +p1 ./amr_1d_random)
	$(call run, +p2 ./amr_1d_random)
	$(call run, +p3 ./amr_1d_random)
	$(call run, +p4 ./amr_1d_random)

testp: amr_1d_random
	$(call run, +p$(P) ./amr_1d_random)

smptest: amr_1d_random
	$(call run, +p2 ./amr_1d_random)
	$(call run, +p3 ./amr_1d_random)
	$(call run, +p4 ./amr_1d_random)

clean:
	rm -rf *.decl.h *.def.h amr_1d_random charmrun
