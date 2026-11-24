-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
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
# The parameter search of the GPU greedy-refine balancer (concurrent mode).
LBS += GreedyRefineCentralGPULB,+LBGreedyRefineTolerance,0:-expect,improve
# Load predictor of CentralLB: stores 2 steps, builds the model, then updates
# it. The strategy balances the predicted loads of the next step, so the load
# is uniform, where those equal the loads the check uses.
LBS += MetisLB,+LBPredictor,+LBPredictorDelay,2,+LBPredictorWindow,2:-expect,improve,-tol,1.5,-steps,5,-n,16,-nonmig,0,-profile,uniform
# Summary and debug printing of CentralLB and TreeLB.
LBS += GreedyRefineCentralGPULB,+LBPrintSummary,+LBDebug,2:-expect,improve
LBS += GreedyLB,+LBPrintSummary,+LBDebug,2:-expect,improve
# Several +balancer options: one balancer per step, the last one kept, or in
# turn with +LBLoop.
LBS += GreedyLB,+balancer,RotateLB:-expect,improve/rotate
LBS += MetisLB,+balancer,DummyLB:-expect,improve/dummy,-tol,1.5
LBS += RotateLB,+balancer,DummyLB,+LBLoop:-expect,rotate/dummy/rotate
# Manual load balancing: TurnManualLBOn on every PE, CkStartLB from Main once
# every element has called AtSync.
LBS += GreedyLB:-expect,improve,-manual,1
LBS += MetisLB:-expect,improve,-tol,1.5,-manual,1

comma = ,
lbargs = $(subst $(comma), ,$(word $(1),$(subst :, ,$(2))))
define lbline
	$(call run, ./lbsweep +p4 $(call lbargs,2,$(1)) $(EXTRA) +balancer $(call lbargs,1,$(1)))

endef

# The LBDump run stops the program (CkExit) right after writing the step 1
# data, and the LBSim run stops after simulating that step, so those two runs
# check that the file was written and read rather than the placement.
#
# 2groups.json splits the processes into two process groups; the root passes
# load tokens between them. It needs two processes, so it is left out where
# RUN_PROCS is empty (the Windows job sets RUN_PROCS=).
ifeq ($(CMK_GDIR),reconverse)
RUN_PROCS = TESTRUN_PROCS=2
endif

test: lbsweep
	$(foreach e,$(LBS),$(call lbline,$(e)))
	$(call run, ./lbsweep +p4 -profile ramp -nonmig 0 -expect improve +balancer GreedyLB)
	$(call run, ./lbsweep +p4 -profile uniform -nonmig 0 -expect improve -tol 1.5 +balancer RefineLB)
	$(call run, ./lbsweep +p4 -n 3 -nonmig 0 -expect improve +balancer GreedyLB)
	$(call run, ./lbsweep +p4 -n 3 -expect improve +balancer GreedyLB)
	$(call run, ./lbsweep +p1 -expect dummy +balancer GreedyLB)
	$(call run, ./lbsweep +p1 -expect dummy +balancer MetisLB)
	$(call run, ./lbsweep +p1 -expect dummy +balancer DistributedLB)
	rm -f lbdump.* lbsim.out
	$(call run, ./lbsweep +p4 -expect improve -tol 1.5 +balancer MetisLB +LBDump 1 +LBDumpSteps 1 +LBDumpFile lbdump)
	test -s lbdump.1
	$(call run, ./lbsweep +p4 -expect improve -tol 1.5 +balancer MetisLB +LBSim 1 +LBSimSteps 1 +LBDumpFile lbdump) > lbsim.out 2>&1 || { cat lbsim.out; exit 1; }
	cat lbsim.out
	grep -q "LBSim: Simulation of load balancing step 1 done" lbsim.out
	rm -f lbdump.* lbsim.out
ifneq ($(RUN_PROCS),)
	$(RUN_PROCS) $(call run, ./lbsweep +p4 -expect improve -tol 1.5 -localfreq 2 +balancer TreeLB +TreeLBFile 2groups.json +LBDebug 1)
endif

smptest: all

clean:
	rm -f *.decl.h *.def.h *.o lbsweep charmrun lbdump.* lbsim.out
