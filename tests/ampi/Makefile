DIRS=megampi stacksize migration
#DIRS=megampi stacksize chkpt 

all:
	for d in $(DIRS); do \
		(cd $$d && $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		(cd $$d && $(MAKE) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(DIRS); do \
		(cd $$d && $(MAKE) bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

syncfttest causalfttest:
	for d in jacobi3d; do \
        (cd $$d && $(MAKE) $@ OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
    done

mpisyncfttest mlogfttest:

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
