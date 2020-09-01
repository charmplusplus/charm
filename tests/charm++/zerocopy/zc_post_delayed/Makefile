-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: zc_post_delayed

OBJS = zc_post_delayed.o

zc_post_delayed: $(OBJS)
	$(CHARMC) -language charm++ -o zc_post_delayed $(OBJS)

cifiles: zc_post_delayed.ci
	$(CHARMC)  zc_post_delayed.ci
	touch cifiles

zc_post_delayed.o: zc_post_delayed.C cifiles
	$(CHARMC) -c zc_post_delayed.C

test: all
	$(call run, +p1 ./zc_post_delayed)
	$(call run, +p4 ./zc_post_delayed)
	$(call run, +p6 ./zc_post_delayed)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p4 ./zc_post_delayed +noCMAForZC)
	$(call run, +p6 ./zc_post_delayed +noCMAForZC)
endif

testp: all
	$(call run, +p$(P) ./zc_post_delayed )
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p$(P) ./zc_post_delayed +noCMAForZC)
endif

smptest: all
	$(call run, +p2 ./zc_post_delayed ++ppn 2)
	$(call run, +p4 ./zc_post_delayed ++ppn 2)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p2 ./zc_post_delayed +noCMAForZC ++ppn 2)
	$(call run, +p4 ./zc_post_delayed +noCMAForZC ++ppn 2)
endif

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f zc_post_delayed charmrun cifiles
