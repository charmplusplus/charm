TEST_TOOLS=$(shell CHARMINC=../../../../include/; if test -f ./conv-config.sh; then . ./conv-config.sh; echo $$BGP_FLOOR; fi )

DIRS= loadlog buildlog

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

ifeq "$(TEST_TOOLS)" ""
test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' || exit 1) || exit 1; \
	done
else
test:
endif

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
