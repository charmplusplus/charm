CHARMC=../../../bin/charmc $(OPTS)

all:
	cd master-slave; make all OPTS='$(OPTS)'
	cd commlib; make all OPTS='$(OPTS)'
	cd packlib; make all OPTS='$(OPTS)'
	cd metis; make all OPTS='$(OPTS)'
	cd ddt; make all OPTS='$(OPTS)'

clean:
	cd master-slave; make clean
	cd commlib; make clean
	cd packlib; make clean
	cd metis; make clean
	cd ddt; make clean
