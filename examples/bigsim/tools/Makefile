TEST_TOOLS=$(shell CHARMINC=../../../../include/; if test -f ./conv-config.sh; then . ./conv-config.sh; echo $$BGP_FLOOR; fi )

DIRS= loadlog buildlog

all:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

ifeq "$(TEST_TOOLS)" ""
test:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)' || exit 1) || exit 1; \
	done
else
test:
endif

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
