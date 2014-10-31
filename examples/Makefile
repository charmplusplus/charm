DIRS=charm++ converse ampi armci bigsim collide

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

test-fem:
	$(MAKE) -C fem test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

bgtest:
	for d in charm++ ampi armci; do \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do \
		($(MAKE) -C $$d clean OPTS='$(OPTS)'); \
	done
	rm -f TAGS #*#
	rm -f core *~
