-include ../../include/conv-mach-opt.mak

DIRS = \
  commbench \
  cthtest \
  machinetest \
  pingpong \
  randomttl \
  kNeighbors \

# Not applicable on reconverse: commbench is C, and reconverse's converse.h
# declares the Converse API without CLINKAGE, so libreconverse exports it with
# C++ linkage and C objects do not link (tracked in charm #4035).
ifeq ($(CMK_GDIR),reconverse)
RECONVERSE_NA = \
  commbench \

endif

DIRS := $(filter-out $(RECONVERSE_NA),$(DIRS))

TESTDIRS = $(DIRS)

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

testp: $(foreach i,$(TESTDIRS),testp-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'

$(foreach i,$(DIRS),testp-$i):
	$(MAKE) -C $(subst testp-,,$@) testp OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'
