-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all: ping_all

ping_all:  ping_all.o
	$(CHARMC) ping_all.o -o ping_all -language charm++ -module CommonLBs

cifiles: ping_all.ci
	$(CHARMC) -c ping_all.ci
	touch cifiles

ping_all.o : ping_all.C cifiles
	$(CHARMC) -c ping_all.C

test: all
	$(call run, +p4 ./ping_all)

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun ping_all
