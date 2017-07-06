DIRS	= \
allToAll \
cksequence \
completion \
fib \
groupsection \
hello \
integrate \
integratePUPable \
integrateArray \
jacobi2d-1d-decomposition \
jacobi2d-2d-decomposition \
jacobi3d-2d-decomposition \
kmeans \
leanmd \
load_balancing \
manyToMany \
matmul \
namespace \
piArray \
PUP \
NQueen \
reductions \
ring \
rings \
speeds \
sync_square \
threaded_ring \
TRAM \
zerocopy

BGDIRS=\
hello \
integrate \
integratePUPable \
integrateArray \
load_balancing \
piArray \
PUP \
NQueen \
reductions \
ring \
rings \
speeds

all: 
	for d in $(DIRS); do \
		($(MAKE) -C $$d all OPTS='$(OPTS)' || exit 1) || exit 1; \
	done

test:
	for d in $(DIRS); do \
		($(MAKE) -C $$d test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

bgtest:
	for d in $(BGDIRS); do \
		($(MAKE) -C $$d bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' || exit 1) || exit 1; \
	done

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~
