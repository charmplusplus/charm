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
# priority phase. Each expect_abort run must abort with the CkDyn message in
# ABORT_<name>: a send to a deleted section, a broadcast with an NCPY slot, a
# section id from before +restart (the restart of the +p1 checkpoint), a
# section of a charmxi array, a SECTION callback as checkpoint callback,
# deleting a combined section that a combine waits on, and the reduction
# number of a deleted section, a CkDynRegisterClass call after start-up, and
# shaped contributions of two shapes to one reduction.
ABORT_stalerestart = CkDynSendSection: stale section id from before restart
ABORT_stalesend = send to a deleted section .*stale generation
ABORT_staleredno = CkDynSectionRedNo: .*was deleted
ABORT_zcbcast = slot 2 is an NCPY slot; a message with zero-copy slots is sent point to point only, not as an array broadcast
ABORT_badlatereg = CkDynRegisterClass for class Late after start-up; register it with CkDynRegisterClassLate
ABORT_xisection = a member is an element of class Plain, which is not a dynamic class
ABORT_ckptsection = CkDynCheckpoint: the done callback is a SECTION callback
ABORT_deletewaiting = combine\(s\) named it as a part
ABORT_mixedshape = contributions to one reduction have different shapes, \((2, 3|3, 2)\) and \((2, 3|3, 2)\)

# The run goes through testrun, so a batch launcher starts it like the other
# runs. It must exit nonzero and its output must match ABORT_$(1) (an
# extended regular expression). testrun skips (exit 0, "Skipping the test")
# when the process count does not divide the PE count; that counts as a pass.
# $(1): the case name, $(2): testrun and program options
define expect_abort
	@$(call run, $(2) ) > $(1).log 2>&1; rc=$$?; \
	if grep -q "Skipping the test" $(1).log; then \
	  echo "ckdyn $(1) skipped"; \
	elif [ $$rc -eq 0 ]; then \
	  cat $(1).log; echo "ckdyn $(1): expected a nonzero exit"; exit 1; \
	elif ! grep -qE "$(ABORT_$(1))" $(1).log; then \
	  cat $(1).log; echo "ckdyn $(1): abort message missing"; exit 1; \
	else \
	  echo "ckdyn $(1) passed (exit $$rc)"; \
	fi
endef

test: all
	$(call run, ./ckdyntest +p4 +balancer RotateLB +randomized_msgq -skip priority -requiremoves )
	$(call run, ./ckdyntest +p4 +balancer RotateLB -requiremoves )
	$(call run, ./ckdyntest +p4 +restart ckdyntest.ckpt )
	$(call run, ./ckdyntest +p1 )
	$(call run, ./ckdyntest +p1 +restart ckdyntest.ckpt )
	$(call expect_abort,stalerestart,./ckdyntest +p1 +restart ckdyntest.ckpt -stalerestart)
	$(call expect_abort,stalesend,./ckdyntest +p1 -stalesend -skip pingpong -skip pingpong_inline)
	$(call expect_abort,staleredno,./ckdyntest +p2 -staleredno -skip pingpong -skip pingpong_inline)
	$(call expect_abort,zcbcast,./ckdyntest +p1 -zcbcast -skip pingpong -skip pingpong_inline)
	$(call expect_abort,badlatereg,./ckdyntest +p1 -badlatereg)
	$(call expect_abort,xisection,./ckdyntest +p2 -xisection)
	$(call expect_abort,ckptsection,./ckdyntest +p1 -ckptsection -skip sections -skip pingpong -skip pingpong_inline)
	$(call expect_abort,deletewaiting,./ckdyntest +p2 -deletewaiting -skip pingpong -skip pingpong_inline)
	$(call expect_abort,mixedshape,./ckdyntest +p2 -mixedshape -skip pingpong -skip pingpong_inline)

testp: all
	$(call run, ./ckdyntest +p$(P) +balancer RotateLB )
