DIRS=multicast

all: 
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(DIRS); do \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
