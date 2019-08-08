-include ../../include/conv-mach-opt.mak

DIRS = \
  megacon \

TESTDIRS = $(DIRS)

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)
ifeq ($(CMK_SMP),1)
ifneq ($(CMK_MULTICORE),1)
	make smptest
endif
endif

smptest: $(foreach i,$(TESTDIRS),smptest-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),smptest-$i):
	$(MAKE) -C $(subst smptest-,,$@) smptest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
