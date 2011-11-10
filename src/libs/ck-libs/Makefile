CHARMC=../../bin/charmc $(OPTS)
CHARMINC=.

SIMPLE_DIRS = completion cache sparseContiguousReducer tcharm ampi idxl \
              parmetis multiphaseSharedArrays fem ParFUM \
              ifem armci collide mblock barrier irecv netfem liveViz \
              taskGraph search datatransfer MeshStreamer pose
DIRS = $(SIMPLE_DIRS) pythonCCS

.PHONY: $(DIRS)

all:  $(DIRS)

$(SIMPLE_DIRS) ParFUM-Tops:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ampi:  tcharm
idxl:  ampi
parmetis: ampi
multiphaseSharedArrays:
fem:  idxl ampi parmetis
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
datatransfer: ampi fem

pythonCCS:
	-(CHARMINC=../../../include;. $$CHARMINC/conv-config.sh; \
	if test "$$CMK_BUILD_PYTHON" != ""; then ($(MAKE) -C $@ all OPTS='$(OPTS)' || exit 1); fi)


clean:
	for d in $(DIRS); do (cd $$d && $(MAKE) clean OPTS='$(OPTS)'); done
