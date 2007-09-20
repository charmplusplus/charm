DIRS=simpletest simpleTestVarsize 

all:
	for d in $(DIRS); do \
		(cd $$d; make all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		(cd $$d; make test OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; make clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
