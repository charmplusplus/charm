-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: dynamic_insertion_deletion

dynamic_insertion_deletion: dynamic_insertion_deletion.decl.h dynamic_insertion_deletion.C
	$(CHARMC) dynamic_insertion_deletion.C -o dynamic_insertion_deletion 

dynamic_insertion_deletion.decl.h: dynamic_insertion_deletion.ci
	$(CHARMC) dynamic_insertion_deletion.ci

test: dynamic_insertion_deletion
	$(call run, +p3 ./dynamic_insertion_deletion)

testp: dynamic_insertion_deletion
	$(call run, +p$(P) ./dynamic_insertion_deletion)

smptest: dynamic_insertion_deletion
	$(call run, +p2 ./dynamic_insertion_deletion)
	$(call run, +p3 ./dynamic_insertion_deletion)
	$(call run, +p4 ./dynamic_insertion_deletion)

clean:
	rm -rf *.decl.h *.def.h dynamic_insertion_deletion charmrun
