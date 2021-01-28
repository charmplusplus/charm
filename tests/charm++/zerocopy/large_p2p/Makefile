-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMC=../../../../bin/charmc $(OPTS)

all: large_p2p

OBJS = large_p2p.o

large_p2p: $(OBJS)
	$(CHARMC) -language charm++ -o large_p2p $(OBJS)

cifiles: large_p2p.ci
	$(CHARMC)  large_p2p.ci
	touch cifiles

large_p2p.o: large_p2p.C cifiles
	$(CHARMC) -c large_p2p.C

test: all # Passed parameter is MAX_CMA_RW_COUNT + 6
	$(call run, +p1 ./large_p2p 2147479558 )
	$(call run, +p2 ./large_p2p 2147479558 )
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p2 ./large_p2p 2147479558 +noCMAForZC )
endif

clean:
	rm -f *.decl.h *.def.h conv-host *.o large_p2p charmrun cifiles
