OPTS=-I../ -O3 -g -lpthread

CHARMDIR=$(HOME)/curcvs/charm/net-linux-x86_64-smp-opt
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
CHARMLIB=$(CHARMDIR)/lib
CHARMINC=$(CHARMDIR)/include

all: module
	make install

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


NodeHelper.o: NodeHelper.C NodeHelper.decl.h NodeHelper.h NodeHelperAPI.h
	$(CHARMC) -c NodeHelper.C

install: $(CHARMLIB)/libmoduleNodeHelper.a
	cp NodeHelperAPI.h NodeHelper.h NodeHelper.decl.h NodeHelper.def.h $(CHARMINC)/

