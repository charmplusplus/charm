-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

all: async_arrival

async_arrival: async_arrival.o
	$(CHARMC) -language charm++ -o $@ $< -module CommonLBs

async_arrival.decl.h async_arrival.def.h: async_arrival.ci
	$(CHARMC) $<

async_arrival.o: async_arrival.C async_arrival.decl.h
	$(CHARMC) -c $<

# Arguments: steps, split (0: AtSync, 1: AtSyncStart/AtSyncWait), wait (the
# AtSync argument; 0 is an asynchronous arrival). GreedyRefineCentralGPULB is a
# CentralLB, which counts asynchronous arrivals (TreeLB refuses them), and moves
# objects without a GPU. +LBScatterThreshold 1 makes +LBPeerDecision (which
# +LBAsync forces) scatter the decision down a tree even at 4 PEs, instead of
# broadcasting it. DistributedLB ends its steps on the move ledger under
# +LBSyncResume (which +LBAsync forces) and on arrival counts without it; its
# gossip ends by quiescence, so under +LBAsync it runs with AtSync, where the
# elements stop in AtSyncWait and the application goes quiet (an element that
# keeps polling, as split=1 does here, would hold the step open). The
# runs without +balancer have no balancer at all, so a step finishes inside the
# AtSyncStart call that completes a PE's barrier.
test: async_arrival
	$(call run, ./async_arrival 4 0 0 +p2 +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 0 0 +p4 +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 1 1 +p4 +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 1 1 +p4 +LBAsync +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 1 0 +p4 +LBAsync +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 0 1 +p4 +LBAsync +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 0 0 +p4 +LBPeerDecision +LBScatterThreshold 1 +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 1 1 +p4 +LBAsync +LBScatterThreshold 1 +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 4 0 1 +p4 +balancer DistributedLB)
	$(call run, ./async_arrival 4 0 1 +p4 +LBSyncResume +balancer DistributedLB)
	$(call run, ./async_arrival 4 0 1 +p4 +LBAsync +balancer DistributedLB)
	$(call run, ./async_arrival 4 1 1 1 +p4 +LBAsync +MetaLB +MetaLBGpuTrigger +balancer GreedyRefineCentralGPULB)
	$(call run, ./async_arrival 3 0 1 +p2)
	$(call run, ./async_arrival 3 1 1 +p2)
	$(call run, ./async_arrival 3 1 1 +p2 +LBAsync)

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o async_arrival charmrun
