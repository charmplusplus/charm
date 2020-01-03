-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: dereg_and_nodereg

OBJS = dereg_and_nodereg.o

dereg_and_nodereg: $(OBJS)
	$(CHARMC) -language charm++ -o dereg_and_nodereg $(OBJS)

cifiles: dereg_and_nodereg.ci
	$(CHARMC)  dereg_and_nodereg.ci
	touch cifiles

dereg_and_nodereg.o: dereg_and_nodereg.C cifiles
	$(CHARMC) -c dereg_and_nodereg.C

test: all
	$(call run, +p1 ./dereg_and_nodereg 10)
	$(call run, +p4 ./dereg_and_nodereg 50)
	$(call run, +p6 ./dereg_and_nodereg 100)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p4 ./dereg_and_nodereg 50 +noCMAForZC)
	$(call run, +p6 ./dereg_and_nodereg 100 +noCMAForZC)
endif

smptest: all
	$(call run, +p4 ./dereg_and_nodereg 50 ++ppn 4)
	$(call run, +p6 ./dereg_and_nodereg 100 ++ppn 3)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p4 ./dereg_and_nodereg 50 +noCMAForZC ++ppn 4)
	$(call run, +p6 ./dereg_and_nodereg 100 +noCMAForZC ++ppn 3)
endif

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f dereg_and_nodereg charmrun cifiles
