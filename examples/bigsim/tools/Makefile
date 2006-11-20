DIRS= loadlog buildlog

all:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) OPTS='$(OPTS)' || exit 1) || exit 1; \
	done


test:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
