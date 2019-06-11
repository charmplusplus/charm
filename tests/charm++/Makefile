DIRS = \
  megatest \
  alignment \
  simplearrayhello \
  provisioning \
  load_balancing \
  chkpt \
  delegation \
  queue \
  sdag \
  ckAllocSysMsgTest \
  method_templates \
  demand_creation \
  startupTest \
  topology \
  io \
  sparse \
  reductionTesting \
  partitions \
  charmxi_parsing \
  jacobi3d \
  jacobi3d-sdag \
  zerocopy \

# skip sdag, megatest and commtest
BGDIRS = \
  simplearrayhello \
  load_balancing \
  chkpt \
  delegation \

FTDIRS = \
  jacobi3d \
  jacobi3d-sdag \

MLOGFTDIRS = \
  jacobi3d-sdag \

TESTDIRS = $(filter-out $(FTDIRS),$(DIRS))

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

bgtest: $(foreach i,$(filter $(BGDIRS),$(TESTDIRS)),bgtest-$i)

syncfttest causalfttest mpisyncfttest: $(foreach i,$(filter $(FTDIRS),$(DIRS)),test-$i)

mlogfttest: $(foreach i,$(filter $(MLOGFTDIRS),$(DIRS)),test-$i)

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
