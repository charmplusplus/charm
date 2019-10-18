CDIR := ../..
-include $(CDIR)/include/conv-mach-opt.mak
CHARMC=$(CDIR)/bin/charmc $(OPTS)
CHARMINC=.

SIMPLE_DIRS = completion cache sparseContiguousReducer tcharm ampi idxl \
              multiphaseSharedArrays io \
              armci collide mblock barrier irecv liveViz \
              taskGraph search MeshStreamer NDMeshStreamer pose \
              state_space_searchengine
DIRS = $(SIMPLE_DIRS) pythonCCS
FEM_DIRS = fem ifem datatransfer netfem

.PHONY: $(DIRS) ParFUM netfem

all:  $(DIRS)

$(SIMPLE_DIRS) ParFUM ParFUM-Tops $(FEM_DIRS):
	$(MAKE) -C $@ all OPTS='$(OPTS)'

FEM: $(FEM_DIRS) 

ampi:  tcharm
idxl:  ampi
multiphaseSharedArrays:
fem:  idxl ampi multiphaseSharedArrays
ParFUM:	idxl ampi collide multiphaseSharedArrays
ParFUM-Tops: ParFUM
ifem: fem
armci: ampi
collide: tcharm
mblock: tcharm
barrier:
irecv:
netfem: tcharm
liveViz:
taskGraph:
search:
datatransfer: ampi fem collide
NDMeshStreamer: completion

pythonCCS:
ifneq ($(CMK_BUILD_PYTHON),)
	$(MAKE) -C $@ all OPTS='$(OPTS)'
endif

clean: $(foreach i,$(DIRS),clean-$i)

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'
