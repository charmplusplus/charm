CHARMC=../../bin/charmc $(OPTS)
CHARMINC=.
DIRS=sparseContiguousReducer tcharm ampi idxl parmetis multiphaseSharedArrays fem ifem \
        armci collide mblock barrier \
	multicast irecv netfem liveViz taskGraph search datatransfer

all:  
	for d in $(DIRS); do (cd $$d && $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; done
	-(CHARMINC=../../../include;. $$CHARMINC/conv-config.sh; \
	if test "$$CMK_BUILD_PYTHON" = "1"; then (cd pythonCCS && $(MAKE) all OPTS='$(OPTS)' || exit 1); fi)

idxl:
	(cd idxl; $(MAKE) all OPTS='$(OPTS)')

tcharm:
	(cd tcharm; $(MAKE) all OPTS='$(OPTS)')

ampi:  tcharm
	(cd ampi; $(MAKE) all OPTS='$(OPTS)')

parmetis: ampi
	(cd parmetis; $(MAKE) all OPTS='$(OPTS)')

fem:  idxl ampi parmetis
	(cd fem; $(MAKE) all OPTS='$(OPTS)')

python:
	(cd pythonCCS; $(MAKE) all OPTS='$(OPTS)')

clean:
	for d in $(DIRS); do (cd $$d && make clean OPTS='$(OPTS)'); done
