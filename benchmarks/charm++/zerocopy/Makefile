DIRS = \
  direct_api \
  entry_method_api \
  entry_method_bcast_api \
  entry_method_post_api \
  entry_method_bcast_post_api \
  p2pPingpong \
  bcastPingAll \
  readonlyBcast \

TESTDIRS = $(DIRS)

TESTPDIRS = \
  entry_method_bcast_api \
  entry_method_bcast_post_api \
  bcastPingAll \
  readonlyBcast \

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

testp: $(foreach i,$(TESTPDIRS),testp-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(TESTPDIRS),testp-$i):
	$(MAKE) -C $(subst testp-,,$@) testp OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' P='$(P)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
