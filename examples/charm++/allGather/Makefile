CHARMC = ../../../bin/charmc $(OPTS)
FLAGS = -g -Wall -Wno-unused-variable -Wno-sign-compare -Wno-class-memaccess -O2
TYPE ?= RING

all: build

build:
	$(CHARMC) $(FLAGS) ./user.ci
	$(CHARMC) $(FLAGS) -o allGather -D$(TYPE) ./user.cc -lallGather

test: build
	./charmrun ++local +p19 ./allGather 19 29 7

testp: build
	./charmrun ++local +p$(P) ./allGather $(P) 29 7

clean:
	rm -f *.decl.h *.def.h *.o charmrun allGather

.phony : all build test clean
