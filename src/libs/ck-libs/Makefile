DIRS=commlib tcharm ampi idxl fem ifem armci collide mblock barrier \
	multicast irecv netfem liveViz liveViz3d taskGraph search
CHARMC=../../bin/charmc $(OPTS)

all:
	for d in $(DIRS); do (cd $$d && make all OPTS='$(OPTS)' || exit 1) || exit 1; done

clean:
	for d in $(DIRS); do (cd $$d && make clean OPTS='$(OPTS)'); done
