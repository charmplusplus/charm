-include ../../../common.mk
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: zerocopy_with_qd

OBJS = zerocopy_with_qd.o

zerocopy_with_qd: $(OBJS)
	$(CHARMC) -language charm++ -o zerocopy_with_qd $(OBJS)

cifiles: zerocopy_with_qd.ci
	$(CHARMC)  zerocopy_with_qd.ci
	touch cifiles

zerocopy_with_qd.o: zerocopy_with_qd.C cifiles
	$(CHARMC) -c zerocopy_with_qd.C

test: all
	$(call run, +p1 ./zerocopy_with_qd 60)
	$(call run, +p4 ./zerocopy_with_qd 100)
	$(call run, +p6 ./zerocopy_with_qd 142)
	$(call run, +p4 ./zerocopy_with_qd 100 +noCMAForZC)
	$(call run, +p6 ./zerocopy_with_qd 142 +noCMAForZC)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f zerocopy_with_qd charmrun cifiles
