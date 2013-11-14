DIRS=migration template case anytimeMigration randomMsgCase

all:
	for d in $(DIRS); do \
	    (cd $$d; $(MAKE) all OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
	    (cd $$d; $(MAKE) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

# No test implemented yet
bgtest:

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
