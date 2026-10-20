CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc -O3
all: sendcost
sendcost.decl.h sendcost.def.h: sendcost.ci
	$(CHARMC) sendcost.ci
sendcost.o: sendcost.C sendcost.decl.h sendcost.def.h
	$(CHARMC) -c sendcost.C
sendcost: sendcost.o
	$(CHARMC) -language charm++ -o sendcost sendcost.o
clean:
	rm -f *.o *.decl.h *.def.h sendcost charmrun
