CPP=g++
COPTS=-ansi -pedantic -g
ALL=Array

all: $(ALL)

clobber:
	rm -rf \#*\# *~ *.o $(ALL)

Array: Array.o
	$(CPP) $(COPTS) $^ -o $@

Test: Array.o Test.o
	$(CPP) $(COPTS) $^ -o $@

Array.o: Array.cpp Array.h
Test.o: Array.o Test.cpp

.cpp.o:
	$(CPP) $(COPTS) -c $< -o $@
