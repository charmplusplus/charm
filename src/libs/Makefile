CHARMC=../../bin/charmc $(OPTS)

all: convlibs charmlibs otherlibs

convlibs:
	cd conv-libs; make all OPTS='$(OPTS)'

charmlibs:
	cd ck-libs; make all OPTS='$(OPTS)'

otherlibs:
	cd other; make all OPTS='$(OPTS)'
