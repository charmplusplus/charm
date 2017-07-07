CHARMRUN=./charmrun
-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: check

check: check.o
	$(CHARMC) $(OPTS) -language charm++ -o check check.o 

check.o: check.C check.decl.h
	$(CHARMC) $(OPTS) -language charm++ -o check.o check.C 


check.decl.h: check.ci
	$(CHARMC) $(OPTS) -language charm++ check.ci

test-local: check
	$(call run, ./check +p4 20 )

clean:
	rm -f *.decl.h *.def.h *.o check charmrun
