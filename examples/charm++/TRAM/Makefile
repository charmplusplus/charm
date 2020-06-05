-include ../../../include/conv-mach-opt.mak

DIRS = aggregateRandomAccessArray

#aggregateRandomAccessGroup can be restored once TRAM support for groups is reimplemented
#the non-aggregate API tests are removed since TRAM now exclusively supports the aggregate API

TESTDIRS = $(DIRS)

ifeq ($(CMK_COMPILER),bgxlc)
  TESTDIRS =
endif

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
