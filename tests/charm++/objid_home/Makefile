-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_home

objid_home: objid_home.decl.h objid_home.C
	$(CHARMC) objid_home.C -o objid_home

objid_home.decl.h: objid_home.ci
	$(CHARMC) objid_home.ci

test: objid_home
	$(call run, +p4 ./objid_home)

smptest: objid_home
	$(call run, +p4 ./objid_home)

clean:
	rm -rf *.decl.h *.def.h objid_home charmrun
