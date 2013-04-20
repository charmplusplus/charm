DIRS= emulator
# Disabled due to host/target incompatibility described in bug #89
#DIRS += tools

BUILD_TRACING=$(shell CHARMINC=../../tmp; if test -f $$CHARMINC/conv-config.sh; then . $$CHARMINC/conv-config.sh; echo $$CMK_TRACE_ENABLED; fi )

ifneq "$(BUILD_TRACING)" "0"
DIRS += sdag
endif

all:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in sdag; do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
