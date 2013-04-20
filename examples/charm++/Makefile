DIRS	= \
AllReduce \
allToAll \
cksequence \
completion \
fib \
hello \
integrate \
integrate2 \
integrateArray \
jacobi2d \
leanmd \
load_balancing \
manyToMany \
matmul \
namespace \
piArray \
PUP \
queens \
RedExample \
ring \
rings \
speeds \
sync_square \
threaded_ring \
typed_reduction

BGDIRS=\
AllReduce \
hello \
integrate \
integrate2 \
integrateArray \
jacobi2d \
load_balancing \
piArray \
PUP \
queens \
RedExample \
ring \
rings \
speeds

all: 
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(BGDIRS); do \
		(cd $$d; $(MAKE) bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do (cd $$d; $(MAKE) clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
