CHARMC ?= charmc
FLAGS = -g -Wall -O2
TYPE ?= RING
ALLGATHER_DIR = ../../../src/libs/ck-libs/allGather
EXAMPLE_DIR=.


allGather:
	$(CHARMC) $(ALLGATHER_DIR)/allGather.ci
	@mv allGather.decl.h $(ALLGATHER_DIR)/allGather.decl.h
	@mv allGather.def.h $(ALLGATHER_DIR)/allGather.def.h
	$(CHARMC) -c $(ALLGATHER_DIR)/allGather.cc
	@echo "Built allGather Library"


collectiveSim: allGather


build-simulation: allGather
	$(CHARMC) $(EXAMPLE_DIR)/user.ci
	$(CHARMC) -D$(TYPE) -I$(ALLGATHER_DIR) $(EXAMPLE_DIR)/user.cc
	$(CHARMC) -o sim user.o allGather.o
	@rm -f user.o allGather.o
	@echo "Built simulation"


test-simulation: build-simulation
	./charmrun ++local +p12 ./sim 12 29 10 10


clean:
	rm -f *.decl.h *.def.h *.o charmrun sim


.phony : allGather build-simulation run-simulation clean
