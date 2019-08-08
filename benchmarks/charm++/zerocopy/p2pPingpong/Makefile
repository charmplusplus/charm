-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMC=../../../../bin/charmc $(OPTS)

all:	megaZCPingpong

OBJS = megaZCPingpong.o

megaZCPingpong: $(OBJS)
	$(CHARMC) -language charm++ -o megaZCPingpong $(OBJS)

cifiles: megaZCPingpong.ci
	$(CHARMC)  megaZCPingpong.ci
	touch cifiles

megaZCPingpong.o: megaZCPingpong.C cifiles
	$(CHARMC) -c megaZCPingpong.C

test: all
	$(call run, +p1 ./megaZCPingpong 1024 8192 2 1 1)
	$(call run, +p2 ./megaZCPingpong 1024 8192 2 1 1)
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p2 ./megaZCPingpong 1024 8192 2 1 1 +noCMAForZC)
endif

test-bench: all
	$(call run, ./megaZCPingpong +p1)
	$(call run, ./megaZCPingpong +p2)
ifeq ($(CMK_USE_CMA),1)
	$(call run, ./megaZCPingpong +p2 +noCMAForZC)
endif

clean:
	rm -f *.decl.h *.def.h conv-host *.o megaZCPingpong charmrun cifiles
