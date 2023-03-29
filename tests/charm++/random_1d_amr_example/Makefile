-include ../../../common.mk
CHARMC	= ../../../netlrts-linux-x86_64/bin/charmc $(OPTS)

all: random_amr

random_amr: random_amr.decl.h random_amr.C
	$(CHARMC) random_amr.C -o random_amr 

random_amr.decl.h: random_amr.ci
	$(CHARMC) random_amr.ci

test: random_amr
	$(call run, +p1 ./random_amr)
	$(call run, +p2 ./random_amr)
	$(call run, +p3 ./random_amr)
	$(call run, +p4 ./random_amr)
	$(call run, +p5 ./random_amr)

testp: random_amr
	$(call run, +p$(P) ./random_amr)

smptest: random_amr
	$(call run, +p2 ./random_amr)
	$(call run, +p3 ./random_amr)
	$(call run, +p4 ./random_amr)

clean:
	rm -rf *.decl.h *.def.h random_amr charmrun
