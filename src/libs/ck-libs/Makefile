DIRS=tcharm armci collide netfem fem ampi mblock barrier \
	multicast irecv liveViz liveViz3d
CHARMC=../../bin/charmc $(OPTS)

all:
	for d in $(DIRS); do (cd $$d && make all OPTS='$(OPTS)' || exit 1) || exit 1; done

clean:
	for d in $(DIRS); do (cd $$d && make clean OPTS='$(OPTS)'); done
