-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS) $(MOPTS) 

OBJS = AllReduce.o

all: AllReduce

AllReduce: $(OBJS)
	$(CHARMC) -language charm++ -o AllReduce $(OBJS) 

AllReduce.decl.h: AllReduce.ci
	$(CHARMC)  AllReduce.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o AllReduce charmrun *.log *.sum *.sts charmrun.exe AllReduce.exe AllReduce.pdb AllReduce.ilk

AllReduce.o: AllReduce.C AllReduce.decl.h AllReduce.h
	$(CHARMC) -c AllReduce.C

test: all
	$(call run, ./AllReduce 20 100000 +p4 )

bgtest: all
	$(call run, ./AllReduce 20 100000 +p4 +x2 +y2 +z2 )
