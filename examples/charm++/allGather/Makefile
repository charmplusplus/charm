CHARMC ?= charmc
FLAGS = -g -Wall -Wno-unused-variable -Wno-sign-compare -Wno-class-memaccess -O2
TYPE ?= RING

build:
	$(CHARMC) $(FLAGS) ./user.ci
	$(CHARMC) $(FLAGS) -o sim -D$(TYPE) ./user.cc -lallGather

test: build
	./charmrun ++local +p19 ./sim 19 29 10 1

clean:
	rm -f *.decl.h *.def.h *.o charmrun sim

.phony : build test clean
