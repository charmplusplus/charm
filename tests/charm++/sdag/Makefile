CHARMRUN=../../../bin/./charmrun
-include ../../common.mk
DIRS=migration template case anytimeMigration randomMsgCase

all:
	for d in $(DIRS); do \
	    ($(MAKE) -C $$d all OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

test-local:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

# No test implemented yet
bgtest:

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
