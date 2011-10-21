OPTS=-I../ -O3 -g -lpthread
CHARMC=$(HOME)/curcvs/charm/net-linux-x86_64-smp-production/bin/charmc $(OPTS)
CHARMLIB=$(HOME)/curcvs/charm/net-linux-x86_64-smp-production/lib
all: module

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun *.log *.sum *.sts

test: all
	./charmrun ./hello +p4 10

bgtest: all
	./charmrun ./hello +p4 10 +x2 +y2 +z2 +cth1 +wth1

module: $(CHARMLIB)/libmoduleNodeHelper.a

$(CHARMLIB)/libmoduleNodeHelper.a: NodeHelper.o
	$(CHARMC)  -o $(CHARMLIB)/libmoduleNodeHelper.a NodeHelper.o


NodeHelper.decl.h: NodeHelper.ci
	$(CHARMC)  NodeHelper.ci


NodeHelper.o: NodeHelper.C NodeHelper.decl.h
	$(CHARMC) -c NodeHelper.C

