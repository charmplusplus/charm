-include ../../common.mk
CHARMDIR = ../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: pup_buffer

OBJS = pup_buffer.o

pup_buffer: $(OBJS)
	$(CHARMC) -language charm++ -o pup_buffer $(OBJS) -module CommonLBs

cifiles: pup_buffer.ci
	$(CHARMC)  pup_buffer.ci
	touch cifiles

pup_buffer.o: pup_buffer.C cifiles
	$(CHARMC) -c pup_buffer.C

test: all
	$(call run, +p1 ./pup_buffer +balancer RotateLB)
	$(call run, +p4 ./pup_buffer +balancer RotateLB)

smptest: all
	$(call run, +p2 ./pup_buffer ++ppn 2 +balancer RotateLB)
	$(call run, +p4 ./pup_buffer ++ppn 2 +balancer RotateLB)
	$(call run, +p2 ./pup_buffer +noCMAForZC ++ppn 2 +balancer RotateLB)
	$(call run, +p4 ./pup_buffer +noCMAForZC ++ppn 2 +balancer RotateLB)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f pup_buffer charmrun cifiles
