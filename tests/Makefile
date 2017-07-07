CHARMRUN=../bin/./charmrun
-include common.mk
# Temporary hack that needs to be removed / replaced
-include ../tmp/buildopts.mk
OPTS += $(OPTSATBUILDTIME)

DIRS=util charm++

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

test-local:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in charm++ ampi; do \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

syncfttest causalfttest mpisyncfttest mlogfttest:
	for d in charm++ ampi; do \
		($(MAKE) -C $$d $@ OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test-converse:
	$(MAKE) -C converse test OPTS='$(OPTS)'

test-charm:
	$(MAKE) -C charm++ test OPTS='$(OPTS)'

test-AMPI:
	$(MAKE) -C ampi test OPTS='$(OPTS)'

clean:
	for d in $(DIRS); do \
		($(MAKE) -C $$d clean OPTS='$(OPTS)'); \
	done
	rm -f TAGS #*#
	rm -f core *~

