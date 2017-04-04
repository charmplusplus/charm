DIRS=3darray fancyarray \
	darray dgroup \
	group local \
	stl_array sdag \
	4darray nodegroup

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)'  TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(DIRS); do \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
