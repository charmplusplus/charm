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
# +messageBufferingThreshold 0 makes every message "large": a message for an id
# with no element is held where it lands and the location asked for, instead of
# being forwarded to the home (phase 7's second path).
test: objid_insert
	$(call run, ./objid_insert +p4)
	$(call run, ./objid_insert +p4 -b)
	$(call run, ./objid_insert +p4 -n)
	$(call run, ./objid_insert +p4 -b -n)
	$(call run, ./objid_insert +p4 +messageBufferingThreshold 0)
	$(call run, ./objid_insert +p4 -n +messageBufferingThreshold 0)
	$(call run, ./objid_insert +p4 -b +messageBufferingThreshold 0)

smptest: objid_insert
	$(call run, ./objid_insert +p4)
	$(call run, ./objid_insert +p4 -b)
	$(call run, ./objid_insert +p4 -n)
	$(call run, ./objid_insert +p4 -b -n)
	$(call run, ./objid_insert +p4 +messageBufferingThreshold 0)
	$(call run, ./objid_insert +p4 -n +messageBufferingThreshold 0)
	$(call run, ./objid_insert +p4 -b +messageBufferingThreshold 0)

clean:
	rm -rf *.decl.h *.def.h objid_insert charmrun
