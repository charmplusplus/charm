DIRS=comlib commlib_stream pipeBroadcast

.PHONY: test $(DIRS)

all:
	for d in $(DIRS); do \
                (cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
        done

test: comlib commlib_stream pipeBroadcast

comlib: 
	$(MAKE) -C $@ test OPTS='$(OPTS)'

commlib_stream:
	$(MAKE) -C $@ test OPTS='$(OPTS)'

pipeBroadcast: 
	$(MAKE) -C $@ test OPTS='$(OPTS)'

bgtest:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) bgtest OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
