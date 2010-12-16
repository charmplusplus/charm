CHARMC=../../../bin/charmc $(OPTS)

OBJS = LB_Test.o

all: LB_Test LB_Test_LB

LB_Test: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Test $(OBJS)

LB_Test_LB: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Test_LB -module CommonLBs $(OBJS)

LB_Test.decl.h: LB_Test.ci
	$(CHARMC)  LB_Test.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o LB_Test LB_Test_LB charmrun *~ *.core

LB_Test.o: LB_Test.C LB_Test.decl.h
	$(CHARMC) -c LB_Test.C
