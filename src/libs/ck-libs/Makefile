CHARMC=../../bin/charmc $(OPTS)
DIRS=commlib sparseContiguousReducer tcharm ampi idxl fem ifem \
        armci collide mblock barrier \
	multicast irecv netfem liveViz liveViz3d taskGraph search datatransfer

all:  
	for d in $(DIRS); do (cd $$d && $(MAKE) all OPTS='$(OPTS)' || exit 1) || exit 1; done

idxl:
	(cd idxl; $(MAKE) all OPTS='$(OPTS)')
	
tcharm:
	(cd tcharm; $(MAKE) all OPTS='$(OPTS)')

ampi:  tcharm
	(cd ampi; $(MAKE) all OPTS='$(OPTS)')

fem:  idxl ampi
	(cd fem; $(MAKE) all OPTS='$(OPTS)')

clean:
	for d in $(DIRS); do (cd $$d && make clean OPTS='$(OPTS)'); done
