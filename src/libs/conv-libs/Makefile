CHARMC=../../../bin/charmc $(OPTS)

all:
	cd master-slave; $(MAKE) all OPTS='$(OPTS)'
	cd commlib; $(MAKE) OPTS='$(OPTS)'
	cd packlib; $(MAKE) all OPTS='$(OPTS)'
	cd metis; $(MAKE) all OPTS='$(OPTS)'
	cd ddt; $(MAKE) all OPTS='$(OPTS)'

clean:
	cd master-slave; make clean
	cd commlib; make clean
	cd packlib; make clean
	cd metis; make clean
	cd ddt; make clean
