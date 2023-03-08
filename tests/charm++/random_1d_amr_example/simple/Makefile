-include ../../../common.mk
CHARMC	= ../../../../bin/charmc $(OPTS)

all: random_amr_simple

random_amr_simple: random_amr_simple.decl.h random_amr_simple.C
	$(CHARMC) random_amr_simple.C -o random_amr_simple 

random_amr_simple.decl.h: random_amr_simple.ci
	$(CHARMC) random_amr_simple.ci

test: random_amr_simple
	$(call run, +p1 ./random_amr_simple)
	$(call run, +p2 ./random_amr_simple)
	$(call run, +p3 ./random_amr_simple)
	$(call run, +p4 ./random_amr_simple)
	$(call run, +p5 ./random_amr_simple)

testp: random_amr_simple
	$(call run, +p$(P) ./random_amr_simple)

smptest: random_amr_simple
	$(call run, +p2 ./random_amr_simple)
	$(call run, +p3 ./random_amr_simple)
	$(call run, +p4 ./random_amr_simple)

clean:
	rm -rf *.decl.h *.def.h random_amr_simple charmrun
