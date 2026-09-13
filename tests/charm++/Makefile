-include ../../include/conv-mach-opt.mak

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
  io_read \
  sparse \
  reductionTesting \
  partitions \
  charmxi_parsing \
  jacobi3d \
  jacobi3d-sdag \
  zerocopy \
  within_node_bcast \
  longIdle \
  periodic \
  bombard \
  varTRAM \
  dynamic_insertion \
  dynamic_insertion_deletion \
  amr_1d_random \
  amr_1d_simple \
  anytime_migration \
  anytime_bcastred \
  qd \

FTDIRS = \
  jacobi3d \
  jacobi3d-sdag \

# Not applicable on reconverse: queue tests classic's Cqs API, which the
# reconverse build does not compile (reconverse has its own queue and its own
# test for it); longIdle needs CcdCallOnConditionOnPE, which reconverse does
# not provide.
ifeq ($(CMK_GDIR),reconverse)
RECONVERSE_NA = \
  queue \
  longIdle \

endif

TESTDIRS = $(filter-out $(FTDIRS) $(RECONVERSE_NA),$(DIRS))

NONSCALEDIRS = \
  alignment \
  queue \
  sdag \
  sparse \
  charmxi_parsing \
  anytime_migration \

TESTPDIRS = $(filter-out $(NONSCALEDIRS),$(TESTDIRS))

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)
ifeq ($(CMK_SMP),1)
ifneq ($(CMK_MULTICORE),1)
	make smptest
endif
endif

testp: $(foreach i,$(TESTPDIRS),testp-$i)

smptest: $(foreach i,$(TESTDIRS),smptest-$i)

syncfttest mpisyncfttest: $(foreach i,$(filter $(FTDIRS),$(DIRS)),test-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(TESTPDIRS),testp-$i):
	$(MAKE) -C $(subst testp-,,$@) testp OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),smptest-$i):
	$(MAKE) -C $(subst smptest-,,$@) smptest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
