# for bigsim only
DIRS=comlib commlib_stream
# BROKEN: pipeBroadcast

.PHONY: test $(DIRS)

# BROKEN: commlib_stream
test: comlib 

comlib: 
	$(MAKE) -C $@ test OPTS='$(OPTS)'

commlib_stream:
	$(MAKE) -C $@ test OPTS='$(OPTS)'

bgtest:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) bgtest OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
