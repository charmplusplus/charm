-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) $(MOPTS) 
#CHARMC=$(HOME)/charm/bin/charmc $(OPTS) $(MOPTS) 


OBJS = ckAllocSysMsgTest.o


all: ckAllocSysMsgTest

ckAllocSysMsgTest: $(OBJS)
	$(CHARMC) -language charm++ -o ckAllocSysMsgTest $(OBJS)

ckAllocSysMsgTest.decl.h: ckAllocSysMsgTest.ci
	$(CHARMC)  ckAllocSysMsgTest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o ckAllocSysMsgTest charmrun *.log *.sum *.sts

ckAllocSysMsgTest.o: ckAllocSysMsgTest.C ckAllocSysMsgTest.decl.h ckAllocSysMsgTest.h
	$(CHARMC) -c ckAllocSysMsgTest.C

test: all
	$(call run, ./ckAllocSysMsgTest 10 10 10 3  +p1 )
	$(call run, ./ckAllocSysMsgTest 30 10 10 3  +p2 )
	$(call run, ./ckAllocSysMsgTest 40  5  2 3  +p2 )
	$(call run, ./ckAllocSysMsgTest 30 10 10 3  +p4 )


bgtest: all
	$(call run, ./ckAllocSysMsgTest 30 10 10 3  +p4 +x2 +y2 +z2)
