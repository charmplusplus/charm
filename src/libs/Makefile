CHARMC=../../bin/charmc $(OPTS)

all: convlibs charmlibs irecvlibs otherlibs

convlibs:
	cd conv-libs; make all OPTS='$(OPTS)'

charmlibs:
	cd ck-libs; make all OPTS='$(OPTS)'

irecvlibs:
	cd irecv; make all OPTS='$(OPTS)'

otherlibs:
	cd other; make all OPTS='$(OPTS)'
