DIRS = randomAccessGroup randomAccessArray aggregateRandomAccessGroup aggregateRandomAccessArray

all:
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_COMPILER" = "bgxlc" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_COMPILER" = "bgxlc" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)'  TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
