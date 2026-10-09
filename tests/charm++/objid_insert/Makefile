-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_insert

objid_insert: objid_insert.decl.h objid_insert.C
	$(CHARMC) objid_insert.C -o objid_insert

objid_insert.decl.h: objid_insert.ci
	$(CHARMC) objid_insert.ci

# Both id kinds, each with every PE sending the createhome demand creations and
# with the home excluded (-n), so a creation must come from a message that
# reaches the home by id.
test: objid_insert
	$(call run, ./objid_insert +p4)
	$(call run, ./objid_insert +p4 -b)
	$(call run, ./objid_insert +p4 -n)
	$(call run, ./objid_insert +p4 -b -n)

smptest: objid_insert
	$(call run, ./objid_insert +p4)
	$(call run, ./objid_insert +p4 -b)
	$(call run, ./objid_insert +p4 -n)
	$(call run, ./objid_insert +p4 -b -n)

clean:
	rm -rf *.decl.h *.def.h objid_insert charmrun
