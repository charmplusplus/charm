-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
SRC=../../../../src
CHARMC=../../../../bin/charmc $(OPTS)

NAME = largedata

OBJS = $(NAME).o

all: large_bcast

large_bcast: $(OBJS)
	$(CHARMC) -language charm++ -o large_bcast $(OBJS)

large_bcast.prj: $(OBJS)
	$(CHARMC) -tracemode projections -language charm++ -o large_bcast.prj $(OBJS)

cifiles: $(NAME).ci
	$(CHARMC) $(NAME).ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o large_bcast charmrun cifiles large_bcast.exe large_bcast.pdb large_bcast.ilk

$(NAME).o: $(NAME).C cifiles
	$(CHARMC) -I$(SRC)/conv-core $<

test: all
	$(call run, ./large_bcast +p4 200000 10)
	$(call run, ./large_bcast +p6 200000 10)
ifeq ($(CMK_USE_CMA),1)
	$(call run, ./large_bcast +p4 200000 10 +noCMAForZC)
	$(call run, ./large_bcast +p6 200000 10 +noCMAForZC)
endif

testp: all
	$(call run, ./large_bcast +p$(P) 200000 10)
ifeq ($(CMK_USE_CMA),1)
	$(call run, ./large_bcast +p$(P) 200000 10 +noCMAForZC)
endif

smptest: all
	$(call run, ./large_bcast +p4 200000 10 ++ppn 2)
	$(call run, ./large_bcast +p4 200000 10 ++ppn 4)
ifeq ($(CMK_USE_CMA),1)
	$(call run, ./large_bcast +p4 200000 10 +noCMAForZC ++ppn 2)
	$(call run, ./large_bcast +p4 200000 10 +noCMAForZC ++ppn 4)
endif

test-bench: all
	$(call run, ./large_bcast +p10)
