-include ../../../../../common.mk
-include ../../../../../../include/conv-mach-opt.mak
CHARMC=../../../../../../bin/charmc $(OPTS)

# nodegroupTest needs at least two processes (logical nodes). On reconverse
# +p<n> through testrun starts one process with n PEs, so ask for two.
ifeq ($(CMK_GDIR),reconverse)
RUN_PROCS = TESTRUN_PROCS=2
endif

all: nodegroupTest

nodegroupTest:  nodegroupTest.o
	$(CHARMC) nodegroupTest.o -o nodegroupTest -language charm++ -module CommonLBs

cifiles: nodegroupTest.ci
	$(CHARMC) -c nodegroupTest.ci
	touch cifiles

nodegroupTest.o : nodegroupTest.C cifiles
	$(CHARMC) -c nodegroupTest.C

test: all
ifeq ($(CMK_MULTICORE),1)
	@echo "Skipping nodegroupTest on MULTICORE builds as it has to be run on an even number of logical nodes"
else
	$(RUN_PROCS) $(call run, +p4 ./nodegroupTest)
ifeq ($(CMK_USE_CMA),1)
	$(RUN_PROCS) $(call run, +p4 ./nodegroupTest +noCMAForZC)
endif
endif

testp: all
ifeq ($(CMK_MULTICORE),1)
	@echo "Skipping nodegroupTest on MULTICORE builds as it has to be run on an even number of logical nodes"
else ifneq (,$(filter %1 %3 %5 %7 %9,$(P))) # if P is odd
	@echo "Skipping nodegroupTest with testp P=$(P) as it has to be run on an even number of logical nodes"
else
	$(RUN_PROCS) $(call run, +p$(P) ./nodegroupTest)
ifeq ($(CMK_USE_CMA),1)
	$(RUN_PROCS) $(call run, +p$(P) ./nodegroupTest +noCMAForZC)
endif
endif

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun nodegroupTest
