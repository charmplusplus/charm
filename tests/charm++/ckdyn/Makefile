-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = ckdyntest.o cbinding.o

all: ckdyntest

ckdyntest: $(OBJS)
	$(CHARMC) -language charm++ -o ckdyntest $(OBJS) -module CommonLBs

ckdyntest.decl.h: ckdyntest.ci
	$(CHARMC) ckdyntest.ci

ckdyntest.o: ckdyntest.C ckdyntest.decl.h cbinding.h
	$(CHARMC) -c ckdyntest.C

cbinding.o: cbinding.C cbinding.h
	$(CHARMC) -c cbinding.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o ckdyntest charmrun *.log *.sum *.sts
	rm -rf ckdyntest.ckpt

# The randomized scheduler queue ignores priorities, so that run skips the
# priority phase. The stale-send run must abort with CkDyn's message.
test: all
	$(call run, ./ckdyntest +p4 +balancer RotateLB +randomized_msgq -skip priority -requiremoves )
	$(call run, ./ckdyntest +p4 +balancer RotateLB -requiremoves )
	$(call run, ./ckdyntest +p4 +restart ckdyntest.ckpt )
	$(call run, ./ckdyntest +p1 )
	$(call run, ./ckdyntest +p1 +restart ckdyntest.ckpt )
	! ./ckdyntest +pe 1 -stalesend -skip pingpong -skip pingpong_inline > stalesend.log 2>&1
	grep -q "send to a deleted section .*stale generation" stalesend.log

testp: all
	$(call run, ./ckdyntest +p$(P) +balancer RotateLB )
