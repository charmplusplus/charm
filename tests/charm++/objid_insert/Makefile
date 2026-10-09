-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_insert

objid_insert: objid_insert.decl.h objid_insert.C
	$(CHARMC) objid_insert.C -o objid_insert

objid_insert.decl.h: objid_insert.ci
	$(CHARMC) objid_insert.ci

test: objid_insert
	$(call run, ./objid_insert +p4)
	$(call run, ./objid_insert +p4 -b)

smptest: objid_insert
	$(call run, ./objid_insert +p4)
	$(call run, ./objid_insert +p4 -b)

clean:
	rm -rf *.decl.h *.def.h objid_insert charmrun
