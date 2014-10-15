# Temporary hack that needs to be removed / replaced
-include ../tmp/buildopts.mk
OPTS += $(OPTSATBUILDTIME)

DIRS=util charm++ converse ampi

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' || exit 1) || exit 1; \
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

test-fem:
	$(MAKE) -C fem test OPTS='$(OPTS)'

clean:
	for d in $(DIRS); do \
		($(MAKE) -C $$d clean OPTS='$(OPTS)'); \
	done
	rm -f TAGS #*#
	rm -f core *~

