-include ../../../../../common.mk
-include ../../../../../../include/conv-mach-opt.mak
CHARMC=../../../../../../bin/charmc $(OPTS)

all: simpleBcast

simpleBcast:  simpleBcast.o
	$(CHARMC) simpleBcast.o -o simpleBcast -language charm++ -module CommonLBs

cifiles: simpleBcast.ci
	$(CHARMC) -c simpleBcast.ci
	touch cifiles

simpleBcast.o : simpleBcast.C cifiles
	$(CHARMC) -c simpleBcast.C

test: all
	$(call run, +p4 ./simpleBcast)
	$(call run, +p6 ./simpleBcast)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p4 ./simpleBcast +noCMAForZC)
	$(call run, +p6 ./simpleBcast +noCMAForZC)
endif

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun simpleBcast
