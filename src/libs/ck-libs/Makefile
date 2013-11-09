CHARMC=../../bin/charmc $(OPTS)
CHARMINC=.

SIMPLE_DIRS = completion cache sparseContiguousReducer tcharm ampi idxl \
              parmetis multiphaseSharedArrays ParFUM io \
              armci collide mblock barrier irecv netfem liveViz \
              taskGraph search MeshStreamer NDMeshStreamer pose \
              state_space_searchengine
DIRS = $(SIMPLE_DIRS) pythonCCS
FEM_DIRS = fem ifem datatransfer

.PHONY: $(DIRS)

all:  $(DIRS)

$(SIMPLE_DIRS) ParFUM-Tops $(FEM_DIRS):
	$(MAKE) -C $@ all OPTS='$(OPTS)'

FEM: $(FEM_DIRS) netfem

ampi:  tcharm
idxl:  ampi
parmetis: ampi
multiphaseSharedArrays:
fem:  idxl ampi parmetis multiphaseSharedArrays
ParFUM:	idxl ampi parmetis collide multiphaseSharedArrays
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
	-(CHARMINC=../../../include;. $$CHARMINC/conv-config.sh; \
	if test "$$CMK_BUILD_PYTHON" != ""; then ($(MAKE) -C $@ all OPTS='$(OPTS)' || exit 1); fi)


clean:
	for d in $(DIRS); do (cd $$d && $(MAKE) clean OPTS='$(OPTS)'); done
