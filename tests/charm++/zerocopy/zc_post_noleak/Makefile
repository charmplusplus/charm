-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: zc_post_noleak

OBJS = zc_post_noleak.o

zc_post_noleak: $(OBJS)
	$(CHARMC) -language charm++ -o zc_post_noleak $(OBJS)

cifiles: zc_post_noleak.ci
	$(CHARMC)  zc_post_noleak.ci
	touch cifiles

zc_post_noleak.o: zc_post_noleak.C cifiles
	$(CHARMC) -c zc_post_noleak.C

test: all
	$(call run, +p2 ./zc_post_noleak 100)
	$(call run, +p4 ./zc_post_noleak 100)

testp: all
	$(call run, +p$(P) ./zc_post_noleak 100)

smptest: all
	$(call run, +p4 ./zc_post_noleak 100 ++ppn 2)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f zc_post_noleak charmrun cifiles
