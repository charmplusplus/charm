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
# priority phase. The stale-send run must abort with CkDyn's message, and so
# must the run that broadcasts an NCPY slot.
test: all
	$(call run, ./ckdyntest +p4 +balancer RotateLB +randomized_msgq -skip priority -requiremoves )
	$(call run, ./ckdyntest +p4 +balancer RotateLB -requiremoves )
	$(call run, ./ckdyntest +p4 +restart ckdyntest.ckpt )
	$(call run, ./ckdyntest +p1 )
	$(call run, ./ckdyntest +p1 +restart ckdyntest.ckpt )
	! ./ckdyntest +pe 1 -stalesend -skip pingpong -skip pingpong_inline > stalesend.log 2>&1
	grep -q "send to a deleted section .*stale generation" stalesend.log
	! ./ckdyntest +pe 1 -zcbcast -skip pingpong -skip pingpong_inline > zcbcast.log 2>&1
	grep -q "slot 2 is an NCPY slot; a message with zero-copy slots is sent point to point only, not as an array broadcast" zcbcast.log
	! ./ckdyntest +pe 2 -xisection > xisection.log 2>&1
	grep -q "a member is an element of class Plain, which is not a dynamic class" xisection.log

testp: all
	$(call run, ./ckdyntest +p$(P) +balancer RotateLB )
