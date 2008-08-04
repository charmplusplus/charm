CHARMC=../../../bin/charmc $(OPTS) $(MOPTS) 
#CHARMC=$(HOME)/charm/bin/charmc $(OPTS) $(MOPTS) 


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
	./charmrun ./startupTest 20 10 0.01 1 $(TESTOPTS) 

bgtest: all
	./charmrun ./startupTest 20 10 0.01 1 +p4 +x2 +y2 +z2
