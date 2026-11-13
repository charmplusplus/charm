-include ../../common.mk
CHARMBASE=../../../
CHARMC=$(CHARMBASE)/bin/ampicxx $(OPTS)

all: lbimbalance lbimbalance-campi

lbimbalance: lbimbalance.C lbimbalance.h
	$(CHARMC) -c lbimbalance.C
	$(CHARMC) -o $@ lbimbalance.o

lbimbalance-campi: lbimbalance-campi.C lbimbalance.h
	$(CHARMC) -std=c++20 -c lbimbalance-campi.C
	$(CHARMC) -o $@ lbimbalance-campi.o

test: lbimbalance lbimbalance-campi
	$(call run, +p2 ./lbimbalance +vp16 +mapping BLOCK_MAP +balancer GreedyLB -steps 40 -lbperiod 10 -a 200)
	$(call run, +p2 ./lbimbalance +vp16 +mapping BLOCK_MAP +balancer RefineLB -pattern spike -steps 40 -lbperiod 10 -a 200 -spikeperiod 20)
	$(call run, +p2 ./lbimbalance-campi +vp16 +mapping BLOCK_MAP +balancer GreedyLB -pattern pivot -steps 40 -lbperiod 10 -a 200)
	$(call run, +p2 ./lbimbalance-campi +vp16 +mapping BLOCK_MAP +balancer RefineLB -pattern spike -steps 40 -lbperiod 10 -a 200 -spikeperiod 20)

testp: lbimbalance
	$(call run, +p$(P) ./lbimbalance +vp$$(( $(P) * 4 )) +mapping BLOCK_MAP +balancer GreedyLB -steps 40 -lbperiod 10 -a 200)

clean:
	rm -f *.o lbimbalance lbimbalance-campi charmrun* *~
