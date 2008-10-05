CHARMC=../../bin/charmc $(OPTS)

all: convlibs charmlibs otherlibs

convlibs:
	cd conv-libs; $(MAKE) all OPTS='$(OPTS)'

charmlibs: convlibs
	cd ck-libs; $(MAKE) all OPTS='$(OPTS)'

otherlibs: charmlibs
	cd other; $(MAKE) all OPTS='$(OPTS)'

clean:
	cd conv-libs; make clean
	cd ck-libs; make clean
	cd other; make clean
