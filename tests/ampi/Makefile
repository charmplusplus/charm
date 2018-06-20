DIRS = \
  megampi \
  stacksize \
  migration \
  intercomm_coll \
  datatype \
  privatization \
  jacobi3d \
#  chkpt \

NOBGDIRS = \
  intercomm_coll \
  privatization \

FTDIRS = \
  jacobi3d

TESTDIRS = $(filter-out $(FTDIRS),$(DIRS))

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

bgtest: $(foreach i,$(filter-out $(NOBGDIRS),$(TESTDIRS)),bgtest-$i)

syncfttest causalfttest: $(foreach i,$(filter $(FTDIRS),$(DIRS)),test-$i)

mpisyncfttest mlogfttest:

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),bgtest-$i):
	$(MAKE) -C $(subst bgtest-,,$@) bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
