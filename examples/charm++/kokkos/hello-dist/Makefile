-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

# Paths need to be set for include and lib folders of Kokkos
# KOKKOS_INC and KOKKOS_LIB respectively
KOKKOS_INC ?= /home/jchoi157/kokkos/build/include
KOKKOS_LIB ?= /home/jchoi157/kokkos/build/lib

OBJS = hello.o

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -L$(KOKKOS_LIB) -lkokkos -lpthread -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC) hello.ci

clean:
	rm -f *.decl.h *.def.h *.o hello charmrun

hello.o: hello.C hello.decl.h
	$(CHARMC) -I$(KOKKOS_INC) -c hello.C

test: all
	$(call run, ./hello +p2 ++local)
