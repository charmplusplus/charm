CHARMC ?= charmc
FLAGS = -g -Wall -O2
TYPE ?= RING
ALLGATHER_DIR=.
EXAMPLE_DIR=./example


allGather:
	$(CHARMC) $(ALLGATHER_DIR)/allGather.ci
	$(CHARMC) -c $(ALLGATHER_DIR)/allGather.cc
	@echo "Built allGather Library"


collectiveSim: allGather


clean:
	rm -f *.decl.h *.def.h *.o charmrun sim


.phony : allGather build-simulation run-simulation clean
