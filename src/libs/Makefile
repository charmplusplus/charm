CHARMC=../../bin/charmc $(OPTS)

all: 
	$(MAKE) convlibs 
	$(MAKE) charmlibs otherlibs

convlibs:
	cd conv-libs; $(MAKE) all OPTS='$(OPTS)'

charmlibs:
	cd ck-libs; $(MAKE) all OPTS='$(OPTS)'

otherlibs:
	cd other; $(MAKE) all OPTS='$(OPTS)'

clean:
	cd conv-libs; make clean
	cd ck-libs; make clean
	cd other; make clean
