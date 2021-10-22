-include ../../common.mk
CHARMC         =  ../../../bin/charmc

all: check

check: check.o
	$(CHARMC) $(OPTS) -language charm++ -o check check.o -module CkMulticast 

check.o: check.C check.decl.h
	$(CHARMC) $(OPTS) -language charm++ -o check.o check.C

check.decl.h: check.ci
	$(CHARMC) $(OPTS) -language charm++ check.ci

test: check
	$(call run, +p4 ./check )

testp: check
	$(call run, +p$(P) ./check )

clean:
	rm -f *.decl.h *.def.h *.o check charmrun
