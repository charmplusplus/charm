-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

all: lbsweep

lbsweep: lbsweep.o
	$(CHARMC) -language charm++ -o $@ $< -module CommonLBs

lbsweep.decl.h lbsweep.def.h: lbsweep.ci
	$(CHARMC) $<

lbsweep.o: lbsweep.C lbsweep.decl.h
	$(CHARMC) -c $<

# One entry per balancer: <+balancer arguments>:<lbsweep options>, with commas
# for spaces. lbsweep.C describes the -expect classes and the tolerances.
LBS += GreedyLB:-expect,improve
LBS += GreedyRefineLB:-expect,improve,-maxmig,0.5
LBS += RefineLB:-expect,improve,-tol,1.5,-maxmig,0.5
LBS += RotateLB:-expect,rotate
LBS += DummyLB:-expect,dummy
LBS += RandCentLB:-expect,random
LBS += DistributedLB:-expect,improve,-tol,2.0
LBS += MetisLB:-expect,improve,-tol,1.5
LBS += RecBipartLB:-expect,improve,-tol,1.5
LBS += GreedyRefineCentralGPULB:-expect,improve
LBS += TreeLB,+TreeLBFile,refineb.json:-expect,improve,-tol,1.5,-maxmig,0.5
LBS += TreeLB,+TreeLBFile,2level.json:-expect,improve,-tol,1.5
LBS += TreeLB,+TreeLBFile,3level.json:-expect,improve,-localfreq,2
LBS += TreeLB,+TreeLBFile,4level.json:-expect,improve

comma = ,
lbargs = $(subst $(comma), ,$(word $(1),$(subst :, ,$(2))))
define lbline
	$(call run, ./lbsweep +p4 $(call lbargs,2,$(1)) $(EXTRA) +balancer $(call lbargs,1,$(1)))

endef

test: lbsweep
	$(foreach e,$(LBS),$(call lbline,$(e)))
	$(call run, ./lbsweep +p4 -profile ramp -nonmig 0 -expect improve +balancer GreedyLB)
	$(call run, ./lbsweep +p4 -profile uniform -nonmig 0 -expect improve -tol 1.5 +balancer RefineLB)
	$(call run, ./lbsweep +p4 -n 3 -nonmig 0 -expect improve +balancer GreedyLB)
	$(call run, ./lbsweep +p4 -n 3 -expect improve +balancer GreedyLB)
	$(call run, ./lbsweep +p1 -expect dummy +balancer GreedyLB)
	$(call run, ./lbsweep +p1 -expect dummy +balancer MetisLB)
	$(call run, ./lbsweep +p1 -expect dummy +balancer DistributedLB)

smptest: all

clean:
	rm -f *.decl.h *.def.h *.o lbsweep charmrun
