#NOTE: to compile the example, the NodeHelper.decl/def.h should exist
USEROPTS=-O0 -g -lpthread
CHARMDIR=$(HOME)/curcvs/charm/net-linux-x86_64-smp-dbg/
CHARMINC=$(CHARMDIR)/include
OPTS=-I$(CHARMINC) $(USEROPTS)
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
CHARMLIB=$(CHARMDIR)/lib


all: hello

hello: hello.o
	$(CHARMC) -language charm++ -o hello hello.o  -module NodeHelper
	$(CHARMC) -language charm++ -o hello.prj hello.o  -tracemode projections -module NodeHelper

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h 
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello hello.prj charmrun *.log *.sum *.sts

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

