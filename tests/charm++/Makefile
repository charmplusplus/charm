DIRS=megatest alignment pingpong simplearrayhello load_balancing chkpt delegation queue sdag ckAllocSysMsgTest method_templates xcastredn demand_creation migrate taskSpawn taskSpawnRecursive streamingAllToAll startupTest kNeighbor topology io sparse reductionTesting

all:
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

# skip sdag, megatest and commtest
bgtest:
	for d in pingpong simplearrayhello load_balancing chkpt delegation; do \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

syncfttest causalfttest mpisyncfttest:
	for d in jacobi3d jacobi3d-sdag; do \
		($(MAKE) -C $$d $@ OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

mlogfttest:
	for d in jacobi3d-sdag; do \
		($(MAKE) -C $$d $@ OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
