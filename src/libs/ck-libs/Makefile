CHARMC=../../bin/charmc $(OPTS)
CHARMINC=.
DIRS=   sparseContiguousReducer tcharm ampi idxl parmetis \
        multiphaseSharedArrays fem ifem ParFUM \
        armci collide mblock barrier \
	irecv netfem liveViz taskGraph search datatransfer pythonCCS

.PHONY: $(DIRS)

all:  $(DIRS)

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

ParFUM:	idxl ampi parmetis collide
	$(MAKE) -C $@ all OPTS='$(OPTS)'

ifem: fem
	$(MAKE) -C $@ all OPTS='$(OPTS)'

armci: tcharm ampi
	$(MAKE) -C $@ all OPTS='$(OPTS)'

collide:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

mblock:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

barrier:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

irecv:
	$(MAKE) -C $@ all OPTS='$(OPTS)'

netfem:
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
