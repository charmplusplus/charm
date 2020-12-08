-include ../../../../../common.mk
-include ../../../../../../include/conv-mach-opt.mak
CHARMC=../../../../../../bin/charmc $(OPTS)

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
	$(call run, +p4 ./nodegroupTest)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p4 ./nodegroupTest +noCMAForZC)
endif
endif

testp: all
ifeq ($(CMK_MULTICORE),1)
	@echo "Skipping nodegroupTest on MULTICORE builds as it has to be run on an even number of logical nodes"
else ifneq (,$(filter %1 %3 %5 %7 %9,$(P))) # if P is odd
	@echo "Skipping nodegroupTest with testp P=$(P) as it has to be run on an even number of logical nodes"
else
	$(call run, +p$(P) ./nodegroupTest)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p$(P) ./nodegroupTest +noCMAForZC)
endif
endif

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun nodegroupTest
