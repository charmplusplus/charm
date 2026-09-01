# GPU-direct (device-to-device zerocopy) examples.
#
# These are reached by the parent examples/charm++/cuda/Makefile's `all` and
# `clean`, but deliberately not by its `test`: every one of them needs real
# GPUs on more than one host to mean anything, so they are run through the
# HPC-site protocol rather than by an automated `make test`. Each subdirectory
# still carries a `test` rule for running it by hand.

DIRS = \
  jacobi2d \
  jacobi2d-imbalance \
  jacobi3d \
  persistent \
  sdag \
  verify \

TESTDIRS = $(DIRS)

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

clean: $(foreach i,$(DIRS),clean-$i)

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(TESTDIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
