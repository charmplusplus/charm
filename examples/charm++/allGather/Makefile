CHARMC = ../../../bin/charmc $(OPTS)
TYPE ?= RING

all: build

build:
	$(CHARMC) ./user.ci
	$(CHARMC) -o allGather -D$(TYPE) ./user.cc -lmoduleallGather

test: build
	$(call run, +p4 ./allGather 4 29 7 )

testp: build
	$(call run, +p$(P) ./allGather $(P) 29 7 )

clean:
	rm -f *.decl.h *.def.h *.o charmrun allGather

.phony : all build test clean
