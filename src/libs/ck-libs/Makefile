CDIR := ../..
-include $(CDIR)/include/conv-mach-opt.mak
CHARMC=$(CDIR)/bin/charmc $(OPTS)
CHARMINC=.

SIMPLE_DIRS = completion cache sparseContiguousReducer tcharm ampi idxl \
              multiphaseSharedArrays io \
              collide mblock allGather barrier irecv liveViz \
              taskGraph search MeshStreamer NDMeshStreamer pose \
              state_space_searchengine
DIRS = $(SIMPLE_DIRS) pythonCCS

.PHONY: $(DIRS) ParFUM

all:  $(DIRS)

$(SIMPLE_DIRS) ParFUM ParFUM-Tops:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ampi:  tcharm
idxl:  ampi
multiphaseSharedArrays:
ParFUM:	idxl ampi collide multiphaseSharedArrays
ParFUM-Tops: ParFUM
collide: tcharm
mblock: tcharm
allGather:
barrier:
irecv:
liveViz:
taskGraph:
search:
NDMeshStreamer: completion

pythonCCS:
ifneq ($(CMK_BUILD_PYTHON),)
	$(MAKE) -C $@ all OPTS='$(OPTS)'
endif

clean: $(foreach i,$(DIRS),clean-$i)

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
