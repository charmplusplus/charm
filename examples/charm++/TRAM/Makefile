DIRS = randomAccessGroup randomAccessArray aggregateRandomAccessGroup aggregateRandomAccessArray

all:
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_USING_BGXLC" = "1" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_USING_BGXLC" = "1" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)'  TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
