-include ../../include/conv-mach-opt.mak

DIRS = \
  allToAll \
  array_map \
  ckcallback \
  ckloop \
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
  openmp \
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
  user-driven-interop \
  zerocopy \

ifeq (1,$(CMK_CAN_LINK_FORTRAN))
# requires fixups
#  DIRS += f90charm
endif

TESTDIRS = $(DIRS)

NONSCALEDIRS = \
  array_map \
  ckcallback \
  cksequence \
  hello \
  namespace \
  PUP \
  NQueen \
  sync_square \

TESTPDIRS = $(filter-out $(NONSCALEDIRS),$(TESTDIRS))

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

testp: $(foreach i,$(TESTPDIRS),testp-$i)

clean: $(foreach i,$(DIRS),clean-$i)
	rm -f TAGS #*#
	rm -f core *~

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),testp-$i):
	$(MAKE) -C $(subst testp-,,$@) testp OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)' P='$(P)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
