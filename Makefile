OPTS=-I../ -O3 -g -lpthread
CHARMC=$(HOME)/charm/net-linux-x86_64-smp-prod/bin/charmc $(OPTS)
CHARMLIB=$(HOME)/charm/net-linux-x86_64-smp-prod/lib
CHARMINC=$(HOME)/charm/net-linux-x86_64-smp-prod/include
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

install: $(CHARMLIB)/libmoduleNodeHelper.a
	cp NodeHelper.h NodeHelper.decl.h NodeHelper.def.h $(CHARMINC)/

