CHARMC=../../bin/charmc $(OPTS)
CHARMINC=.
DIRS=   sparseContiguousReducer tcharm ampi idxl parmetis \
        multiphaseSharedArrays fem ifem ParFUM \
        armci collide mblock barrier cache \
	irecv netfem liveViz taskGraph search datatransfer pythonCCS

.PHONY: $(DIRS)

all:  $(DIRS)

cache:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

sparseContiguousReducer:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

tcharm:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ampi:  tcharm
	$(MAKE) -C $@ all OPTS='$(OPTS)'

idxl:  ampi
	$(MAKE) -C $@ all OPTS='$(OPTS)'

parmetis: ampi
	$(MAKE) -C $@ all OPTS='$(OPTS)'

multiphaseSharedArrays:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

fem:  idxl ampi parmetis
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ParFUM:	idxl ampi parmetis collide multiphaseSharedArrays
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ParFUM-Tops: ParFUM
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ifem: fem
	$(MAKE) -C $@ all OPTS='$(OPTS)'

armci: ampi
	$(MAKE) -C $@ all OPTS='$(OPTS)'

collide: tcharm
	$(MAKE) -C $@ all OPTS='$(OPTS)'

mblock: tcharm
	$(MAKE) -C $@ all OPTS='$(OPTS)'

barrier:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

irecv:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

netfem: tcharm
	$(MAKE) -C $@ all OPTS='$(OPTS)'

liveViz:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

taskGraph:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

search:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

datatransfer: ampi fem
	$(MAKE) -C $@ all OPTS='$(OPTS)'

pythonCCS:
	-(CHARMINC=../../../include;. $$CHARMINC/conv-config.sh; \
	if test "$$CMK_BUILD_PYTHON" != ""; then ($(MAKE) -C $@ all OPTS='$(OPTS)' || exit 1); fi)


clean:
	for d in $(DIRS); do (cd $$d && make clean OPTS='$(OPTS)'); done
