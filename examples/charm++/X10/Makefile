CHARMC=$(HOME)/src/charm/bin/charmc -g

all : X10_test2

X10_test.decl.h : X10_test2.ci
	$(CHARMC) X10_test2.ci

X10_test2.o : X10_test2.C X10_test.decl.h
	$(CHARMC) -c X10_test2.C

X10_test2 : X10_test2.o
	$(CHARMC) -language charm++ -o X10_test2 X10_test2.o 


clean : 
	rm -f a.out *~  X10_test2 charmrun
