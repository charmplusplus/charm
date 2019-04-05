DIRS = \
  allToAll \
  ckcallback \
  cksequence \
  completion \
  fib \
  groupsection \
  hello \
  immediateEntryMethod \
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
  zerocopy \

BGDIRS = \
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
  speeds \

TESTDIRS = $(DIRS)

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

bgtest: $(foreach i,$(filter $(BGDIRS),$(TESTDIRS)),bgtest-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),bgtest-$i):
	$(MAKE) -C $(subst bgtest-,,$@) bgtest OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
