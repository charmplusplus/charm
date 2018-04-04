-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

DIRS = \
  STLPUP \

TESTDIRS = $(DIRS)

all: SimplePUP $(foreach i,$(DIRS),build-$i)

test: test-SimplePUP $(foreach i,$(TESTDIRS),test-$i)

bgtest: bgtest-SimplePUP $(foreach i,$(TESTDIRS),bgtest-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f *.decl.h *.def.h conv-host *.o SimplePUP charmrun

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),bgtest-$i):
	$(MAKE) -C $(subst bgtest-,,$@) bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'

SimplePUP: SimplePUP.o
	$(CHARMC) -language charm++ -o $@ $^

SimplePUP.decl.h: SimplePUP.ci
	$(CHARMC) $<

SimplePUP.o: SimplePUP.C SimplePUP.h SimpleObject.h SimplePUP.decl.h
	$(CHARMC) -c $<

test-SimplePUP: SimplePUP
	$(call run, ./SimplePUP +p4)
	$(call run, ./SimplePUP +p4)

bgtest-SimplePUP: SimplePUP
	$(call run, ./SimplePUP +p4 +x2 +y2 +z1)
