d=threadsafe_hashtable


all: tester

tester: test.cpp
	g++ $< -o $@ -I. -fopenmp -O3 -g

clean:
	-rm tester

tar:
	cd ..; tar czvf $(d).tgz $(d)/Makefile $(d)/*.[ch]* $(d)/osl

