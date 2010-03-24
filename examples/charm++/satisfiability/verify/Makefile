CXX = g++
FLAGS = $(OPTS) 

runverify:verify.o
	$(CXX) -lz -o runverify verify.o
verify.o:verify.C
	$(CXX) $(FLAGS) -o verify.o -c verify.C

clean:
	rm *.o runverify
