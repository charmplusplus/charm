-include ../../../common.mk
CHARMC	= ../../../../bin/charmc $(OPTS)

all: async_arrival

async_arrival: async_arrival.decl.h async_arrival.C
	$(CHARMC) async_arrival.C -o async_arrival -module EveryLB

async_arrival.decl.h: async_arrival.ci
	$(CHARMC) async_arrival.ci

test: async_arrival
	$(call run, +p2 ./async_arrival 4 +balancer GreedyCentralLB)
	$(call run, +p4 ./async_arrival 4 +balancer GreedyCentralLB)

testp: async_arrival
	$(call run, +p$(P) ./async_arrival 4 +balancer GreedyCentralLB)

clean:
	rm -rf *.decl.h *.def.h async_arrival charmrun
