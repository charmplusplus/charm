CHARMRUN=../../bin/./charmrun
-include ../common.mk
DIRS=megampi stacksize migration intercomm_coll
#DIRS=megampi stacksize chkpt 

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test-local:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(DIRS); do \
		if [ "$$d" = "intercomm_coll" ]; then continue; fi; \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

syncfttest causalfttest:
	for d in jacobi3d; do \
        ($(MAKE) -C $$d $@ OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
    done

mpisyncfttest mlogfttest:

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
