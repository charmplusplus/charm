#DIRS=comlib commlib_stream pipeBroadcast
DIRS=commlib_stream pipeBroadcast

.PHONY: test $(DIRS)

all:
	echo "skipping comlib tests"
#	for d in $(DIRS); do \
#                (cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
#        done

#test: comlib commlib_stream pipeBroadcast
test: $(DIRS)

comlib: 
	echo "skipping comlib test"
#	$(MAKE) -C $@ test OPTS='$(OPTS)'

commlib_stream:
	echo "skipping commlib_stream tests"
#	$(MAKE) -C $@ test OPTS='$(OPTS)'

pipeBroadcast: 
	echo "skipping comlib test"
#	$(MAKE) -C $@ test OPTS='$(OPTS)'

bgtest:
	echo "skipping bgtest
#	for d in $(DIRS); do \
#		(cd $$d; $(MAKE) bgtest OPTS='$(OPTS)' || exit 1) || exit 1; \
#	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
