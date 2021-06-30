-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello
	rm -rf log/

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello hello.o $(LIBS) -module CommonLBs

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *~ *.decl.h *.def.h conv-host *.o hello charmrun charmrun.exe hello.exe hello.pdb hello.ilk
	rm -rf log/

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	-rm -fr log
	$(call run, ./hello +p4 +balancer GreedyLB +LBDebug 1 )
	-sync
	$(call run, ./hello +p4 +restart log +balancer GreedyLB +LBDebug 1 )
	$(call run, ./hello +p2 +restart log +balancer GreedyLB +LBDebug 1 )
	-sync
	-rm -fr log
	$(call run, ./hello +p2 )
	-sync
	$(call run, ./hello +p4 +restart log )

testp: all
	-rm -fr log
	$(call run, ./hello +p$(P) )
	-sync
	$(call run, ./hello +p$(P) +restart log )
	$(call run, ./hello +p$$(( $(P) / 2 )) +restart log )
	-sync
	-rm -fr log
	$(call run, ./hello +p$(P) )
	-sync
	$(call run, ./hello +p$(P) +restart log )

smptest: all
	-rm -fr log
	$(call run, ./hello +p4 ++ppn 2 )
	-sync
	$(call run, ./hello +p4 +restart log ++ppn 4 )
	$(call run, ./hello +p2 +restart log ++ppn 2 )
	-sync
	-rm -fr log
	$(call run, ./hello +p2 ++ppn 2)
	-sync
	$(call run, ./hello +p4 +restart log ++ppn 4 )
