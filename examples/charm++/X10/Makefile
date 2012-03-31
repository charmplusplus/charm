CHARMC=../../../bin/charmc $(OPTS)

all : X10_lib.o X10_test2

X10_lib.decl.h : X10_lib.ci
	$(CHARMC) X10_lib.ci

X10_test2.o : X10_test2.C X10_lib.decl.h
	$(CHARMC) -c X10_test2.C

X10_test2 : X10_test2.o X10_lib.o
	$(CHARMC) -language charm++ -o X10_test2 X10_test2.o X10_lib.o

X10_lib.o : X10_lib.C X10_lib.decl.h
	$(CHARMC) -c X10_lib.C


clean : 
	rm -f a.out *~  X10_test2 charmrun X10_lib.o *.decl.h *.def.h
