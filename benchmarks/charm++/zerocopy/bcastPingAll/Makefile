-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

all:	ping_all

OBJS = ping_all.o

ping_all: $(OBJS)
	$(CHARMC) -language charm++ -o ping_all $(OBJS)

cifiles: ping_all.ci
	$(CHARMC)  ping_all.ci
	touch cifiles

ping_all.o: ping_all.C cifiles
	$(CHARMC) -c ping_all.C

test: all
	$(call run, +p1 ./ping_all 1024 8192 2 1 1)
	$(call run, +p2 ./ping_all 1024 8192 2 1 1)
	$(call run, +p2 ./ping_all 1024 8192 2 1 1 +noCMAForZC)

test-bench: all
	$(call run, ./ping_all +p1)
	$(call run, ./ping_all +p2)
	$(call run, ./ping_all +p2 +noCMAForZC)

clean:
	rm -f *.decl.h *.def.h conv-host *.o ping_all charmrun cifiles
