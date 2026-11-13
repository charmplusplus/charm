-include ../../common.mk
CHARMBASE=../../../
CHARMC=$(CHARMBASE)/bin/ampicxx $(OPTS)

all: lbimbalance

lbimbalance: lbimbalance.C
	$(CHARMC) -c lbimbalance.C
	$(CHARMC) -o $@ lbimbalance.o

test: lbimbalance
	$(call run, +p2 ./lbimbalance +vp16 +mapping BLOCK_MAP +balancer GreedyLB -steps 40 -lbperiod 10 -a 200)
	$(call run, +p2 ./lbimbalance +vp16 +mapping BLOCK_MAP +balancer RefineLB -pattern spike -steps 40 -lbperiod 10 -a 200 -spikeperiod 20)

testp: lbimbalance
	$(call run, +p$(P) ./lbimbalance +vp$$(( $(P) * 4 )) +mapping BLOCK_MAP +balancer GreedyLB -steps 40 -lbperiod 10 -a 200)

clean:
	rm -f *.o lbimbalance charmrun* *~
