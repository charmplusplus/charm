-include ../../../include/conv-mach-opt.mak

DIRS = \
# Disabled due to host/target incompatibility described in bug #89
#  tools \

# TODO #2467: eliminate the need for this
ifneq ($(BUILD_CUDA),1)
  DIRS += emulator
endif

ifneq (0,$(CMK_TRACE_ENABLED))
  DIRS += sdag
endif

TESTDIRS = $(DIRS)

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

bgtest: $(foreach i,$(TESTDIRS),bgtest-$i)

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
