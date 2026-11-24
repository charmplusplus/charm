-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_sections

objid_sections: objid_sections.decl.h objid_sections.C
	$(CHARMC) objid_sections.C -o objid_sections -module CkMulticast

objid_sections.decl.h: objid_sections.ci
	$(CHARMC) objid_sections.ci

test: objid_sections
	$(call run, ./objid_sections +p4)

smptest: objid_sections
	$(call run, ./objid_sections +p4)

clean:
	rm -rf *.decl.h *.def.h objid_sections charmrun
