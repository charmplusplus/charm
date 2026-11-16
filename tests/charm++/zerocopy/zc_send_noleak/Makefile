-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: zc_send_noleak

OBJS = zc_send_noleak.o

zc_send_noleak: $(OBJS)
	$(CHARMC) -language charm++ -o zc_send_noleak $(OBJS)

cifiles: zc_send_noleak.ci
	$(CHARMC)  zc_send_noleak.ci
	touch cifiles

zc_send_noleak.o: zc_send_noleak.C cifiles
	$(CHARMC) -c zc_send_noleak.C

test: all
	$(call run, +p2 ./zc_send_noleak 1000)
	$(call run, +p4 ./zc_send_noleak 1000)

testp: all
	$(call run, +p$(P) ./zc_send_noleak 1000)

smptest: all
	$(call run, +p4 ./zc_send_noleak 1000 ++ppn 2)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f zc_send_noleak charmrun cifiles
