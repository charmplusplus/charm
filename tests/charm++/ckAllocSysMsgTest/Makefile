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

define qcbrt
	$(1) := $(2) * 3
endef

CR1 = $(shell expr e(l($(P))/3) )

CR2 = $(shell echo 'e(l($(P))/3)' | bc -l )
#CR3 = $(shell echo 'ceil($(CR2))' | bc -l )
#CR4 = $(shell expr $(CR2) + 1 )

SUM = $(shell expr $(P) + $(P))

NUMCHARES = ""

testp: all
#	$(call run, ./ckAllocSysMsgTest $$(( $(P) ^ (1/3) )) 10 10 3 +p$(P) )
#	$(call run, ./ckAllocSysMsgTest $(eval $(call qcbrt, $(NUMCHARES), $(P))) 10 10 3 +p$(P) )
	$(call run, ./ckAllocSysMsgTest $(CR2) 10 10 3 +p$(P) )

smptest: all
	$(call run, ./ckAllocSysMsgTest 30 10 10 3  +p2 ++ppn 2)
	$(call run, ./ckAllocSysMsgTest 40  5  2 3  +p2 ++ppn 2)
	$(call run, ./ckAllocSysMsgTest 30 10 10 3  +p4 ++ppn 2)
