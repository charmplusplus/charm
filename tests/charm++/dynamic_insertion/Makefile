-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: dynamic_insertion

dynamic_insertion: dynamic_insertion.decl.h dynamic_insertion.C
	$(CHARMC) dynamic_insertion.C -o dynamic_insertion 

dynamic_insertion.decl.h: dynamic_insertion.ci
	$(CHARMC) dynamic_insertion.ci

test: dynamic_insertion
	$(call run, +p1 ./dynamic_insertion)
	$(call run, +p2 ./dynamic_insertion)
	$(call run, +p4 ./dynamic_insertion)

testp: dynamic_insertion
	$(call run, +p$(P) ./dynamic_insertion)

smptest: dynamic_insertion
	$(call run, +p2 ./dynamic_insertion)
	$(call run, +p4 ./dynamic_insertion)

clean:
	rm -rf *.decl.h *.def.h dynamic_insertion charmrun
