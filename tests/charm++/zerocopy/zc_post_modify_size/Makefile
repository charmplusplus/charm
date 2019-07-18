-include ../../../common.mk
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: zc_post_modify_size

OBJS = zc_post_modify_size.o

zc_post_modify_size: $(OBJS)
	$(CHARMC) -language charm++ -o zc_post_modify_size $(OBJS)

cifiles: zc_post_modify_size.ci
	$(CHARMC)  zc_post_modify_size.ci
	touch cifiles

zc_post_modify_size.o: zc_post_modify_size.C cifiles
	$(CHARMC) -c zc_post_modify_size.C

test: all
	$(call run, +p1 ./zc_post_modify_size)
	$(call run, +p4 ./zc_post_modify_size)
	$(call run, +p4 ./zc_post_modify_size +noCMAForZC)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f zc_post_modify_size charmrun cifiles
