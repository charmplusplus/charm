-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: amr_1d_simple

amr_1d_simple: random_amr.decl.h amr_1d_simple.C
	$(CHARMC) amr_1d_simple.C -o amr_1d_simple 

random_amr.decl.h: amr_1d_simple.ci
	$(CHARMC) amr_1d_simple.ci

test: amr_1d_simple
	$(call run, +p2 ./amr_1d_simple)
	$(call run, +p3 ./amr_1d_simple)
	$(call run, +p4 ./amr_1d_simple)

testp: amr_1d_simple
	$(call run, +p$(P) ./amr_1d_simple)

smptest: amr_1d_simple
	$(call run, +p2 ./amr_1d_simple)
	$(call run, +p3 ./amr_1d_simple)
	$(call run, +p4 ./amr_1d_simple)

clean:
	rm -rf *.decl.h *.def.h amr_1d_simple charmrun
