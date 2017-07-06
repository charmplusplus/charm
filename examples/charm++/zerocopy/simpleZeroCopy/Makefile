-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

all: simpleZeroCopy

simpleZeroCopy:  simpleZeroCopy.o
	$(CHARMC) simpleZeroCopy.o -o simpleZeroCopy -language charm++ -module CommonLBs

cifiles: simpleZeroCopy.ci
	$(CHARMC) -c simpleZeroCopy.ci
	touch cifiles

simpleZeroCopy.o : simpleZeroCopy.C cifiles
	$(CHARMC) -c simpleZeroCopy.C

test: all
	$(call run, +p4 ./simpleZeroCopy 200 +balancer RotateLB)
	$(call run, +p4 ./simpleZeroCopy 200 +balancer GreedyLB)

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun simpleZeroCopy
