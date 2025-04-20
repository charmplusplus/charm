CHARMC ?= charmc
FLAGS = -g -Wall -Wno-unused-variable -Wno-sign-compare -O2


allGather:
	$(CHARMC) $(FLAGS) ./allGather.ci
	$(CHARMC) $(FLAGS) -c ./allGather.cc
	@echo "Built allGather Library"


clean:
	rm -f *.decl.h *.def.h *.o


.phony : allGather clean
