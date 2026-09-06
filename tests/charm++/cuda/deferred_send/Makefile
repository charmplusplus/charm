CHARM_DIR = ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc
NVCC = nvcc

all: deferred_send

deferred_send: deferred_send.o delayed_fill.o
	$(CHARMC) -language charm++ -o $@ $^

deferred_send.decl.h: deferred_send.ci
	$(CHARMC) deferred_send.ci

deferred_send.def.h: deferred_send.decl.h

deferred_send.o: deferred_send.C deferred_send.decl.h deferred_send.def.h
	$(CHARMC) -O2 -c deferred_send.C

delayed_fill.o: deferred_send.cu
	$(NVCC) -O2 -c -o $@ $<

# The launcher must place PEs on two physical GPU hosts.
test: all
	$(call run, ./deferred_send +p2)

clean:
	rm -f deferred_send *.o *.decl.h *.def.h charmrun
