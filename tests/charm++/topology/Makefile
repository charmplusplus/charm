-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = rtc.o

all: rtc

rtc: $(OBJS)
	$(CHARMC) -language charm++ -o rtc $(OBJS)

topo.decl.h: topo.ci
	$(CHARMC)  topo.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o rtc charmrun *.log *.sum *.sts

rtc.o: rtc.C topo.decl.h
	$(CHARMC) -c rtc.C

test: all
	$(call run, ./rtc +p4 )

