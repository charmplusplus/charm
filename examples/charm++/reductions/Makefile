DIRS=simple_reduction typed_reduction \
	allReduce

BGDIRS=simple_reduction allReduce

all: 
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)'  TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(BGDIRS); do \
		(cd $$d; $(MAKE) bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
