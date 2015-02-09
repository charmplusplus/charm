-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS) $(MOPTS) 

OBJS = simple_reduction.o

all: simple_reduction

simple_reduction: $(OBJS)
	$(CHARMC) -language charm++ -o simple_reduction $(OBJS)

simple_reduction.decl.h: simple_reduction.ci
	$(CHARMC)  simple_reduction.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o simple_reduction charmrun *.log *.sum *.sts charmrun.exe simple_reduction.exe simple_reduction.pdb simple_reduction.ilk

simple_reduction.o: simple_reduction.C simple_reduction.decl.h simple_reduction.h
	$(CHARMC) -c simple_reduction.C

test: all
	$(call run, ./simple_reduction 20 4.5 10.3 +p4 )

bgtest: all
	$(call run, ./simple_reduction 20 4.5 10.3 +p4 +x2 +y2 +z2 )
