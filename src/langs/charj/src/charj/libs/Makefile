CHARMC:=../../../../../../bin/charmc
CXX:=$(CHARMC)
CC:=$(CHARMC)
LD:=$(CHARMC)
CXXFLAGS=-ansi -pedantic -g
LDFLAGS=-seq -language c++ -lcblas
ALL=Array

all: $(ALL)

test: Array
	./Array

clobber:
	rm -rf \#*\# *~ *.o $(ALL)

Array.o: Array.h
