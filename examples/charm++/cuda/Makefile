DIRS = \
  hello \
  vecadd \
  matmul \
  mempool \
  qdtest \
  busywait \
  stencil2d \
  gpudirect \

# gpudirect is built but not run: the device-to-device examples need real GPUs
# on more than one host to mean anything, so they go through the HPC-site
# protocol rather than an automated `make test`. Building them is still worth
# doing everywhere -- it is what catches an incomplete rename or a header that
# stopped being self-sufficient for user code.
TESTDIRS = \
  hello \
  vecadd \
  matmul \
  mempool \
  qdtest \
  busywait \
  stencil2d \

all: $(foreach i,$(DIRS),build-$i)

test: $(foreach i,$(TESTDIRS),test-$i)

clean: $(foreach i,$(DIRS),clean-$i)

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) OPTS='$(OPTS)'

$(foreach i,$(TESTDIRS),test-$i):
	$(MAKE) -C $(subst test-,,$@) test OPTS='$(OPTS)' TESTOPTS='$(TESTOPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
