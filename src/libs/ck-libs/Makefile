CHARMC=../../bin/charmc $(OPTS)

all:
	cd tcharm; make all OPTS='$(OPTS)'
	cd collide; make all OPTS='$(OPTS)'
	cd netfem; make all OPTS='$(OPTS)'
	cd fem; make all OPTS='$(OPTS)'
	cd ampi; make all OPTS='$(OPTS)'
	cd mblock; make all OPTS='$(OPTS)'
	cd barrier; make all OPTS='$(OPTS)'
	cd multicast; make all OPTS='$(OPTS)'
	cd irecv; make all OPTS='$(OPTS)'

clean:
	cd tcharm; make clean
	cd collide; make clean
	cd netfem; make clean
	cd fem; make clean
	cd ampi; make clean
	cd amr; make clean
	cd mblock; make clean
	cd barrier; make clean
	cd multicast; make clean
	cd irecv; make clean

