CHARMC=../../bin/charmc $(OPTS)

all:
	cd barrier; make all OPTS='$(OPTS)'
	cd irecv; make all OPTS='$(OPTS)'
	cd fem; make -k all OPTS='$(OPTS)'
