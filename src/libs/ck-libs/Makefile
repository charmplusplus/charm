CHARMC=../../bin/charmc $(OPTS)

all:
	cd barrier; make all OPTS='$(OPTS)'
	cd irecv; make all OPTS='$(OPTS)'
	cd fem; make all OPTS='$(OPTS)'
	cd amr; make all OPTS='$(OPTS)'

clean:
	cd barrier; make clean
	cd irecv; make clean
	cd fem; make clean
	cd amr; make clean

