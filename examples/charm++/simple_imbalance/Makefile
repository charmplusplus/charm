CHARMC=../../../bin/charmc $(OPTS)

OBJS = LB_Test.o
PROJ= -tracemode projections -tracemode summary

all: LB_Test LB_Test_LB LB_Test_Proj LB_Test_Proj_LB

LB_Test: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Test $(OBJS)

LB_Test_LB: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Test_LB -module EveryLB $(OBJS)

LB_Test_Proj: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Test_Proj $(PROJ) $(OBJS)

LB_Test_Proj_LB: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Test_Proj_LB -module EveryLB $(PROJ) $(OBJS)

LB_Test.decl.h: LB_Test.ci
	$(CHARMC)  LB_Test.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o LB_Test LB_Test_* charmrun *~ *.core *.projrc *.sts *.log *.sum bgTrace*

LB_Test.o: LB_Test.C LB_Test.decl.h
	$(CHARMC) -c LB_Test.C
