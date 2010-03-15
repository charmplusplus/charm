CHARMC=../../../bin/charmc $(OPTS)

all:
	cd master-slave; $(MAKE) all OPTS='$(OPTS)'
	cd packlib; $(MAKE) all OPTS='$(OPTS)'
#	cd metis; $(MAKE) all OPTS='$(OPTS)'
	cd ddt; $(MAKE) all OPTS='$(OPTS)'

clean:
	cd master-slave; $(MAKE) clean
	cd packlib; $(MAKE) clean
#	cd metis; $(MAKE) clean
	cd ddt; $(MAKE) clean
