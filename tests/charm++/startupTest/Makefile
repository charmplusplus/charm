-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = startupTest.o

all: startupTest

startupTest: $(OBJS)
	$(CHARMC) -language charm++ -o startupTest $(OBJS)

startupTest.decl.h: startupTest.ci
	$(CHARMC)  startupTest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o startupTest charmrun *.log *.sum *.sts

startupTest.o: startupTest.C startupTest.decl.h startupTest.h
	$(CHARMC) -c startupTest.C

test: all
	$(call run, ./startupTest 20  10 0.01   1  +p1)
	$(call run, ./startupTest 200 10 0.0001 1  +p2)
	$(call run, ./startupTest 200 10 0.0001 1  +p4)

bgtest: all
	$(call run, ./startupTest 20 10 0.01 1 +p4 +x2 +y2 +z2)
