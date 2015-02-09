-include ../../common.mk
CXX = ../../../bin/charmc
CXXFLAGS = $(OPTS)

SRC = $(wildcard *.C)
SRCCI = $(wildcard *.ci)
OBJ = $(SRC:.C=.o)
EXEC_NAME = comp.x
DECLS := $(SRCCI:.ci=.decl.h)
DEFS := $(SRCCI:.ci=.def.h)

all: $(EXEC_NAME)

.PHONY: clean

clean:
	rm -f $(OBJ) $(EXEC_NAME) $(DECLS) $(DEFS) charmrun

$(SRC): $(DECLS)

$(EXEC_NAME): $(OBJ)
	$(CXX) -o $@ $^ -module completion

%.decl.h: %.ci
	$(CXX) -c $*.ci

comp.o: comp.C $(DECLS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: all
	$(call run, +p1 ./$(EXEC_NAME) )
	$(call run, +p4 ./$(EXEC_NAME) )
