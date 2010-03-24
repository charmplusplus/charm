CXX = g++
FLAGS = $(OPTS) 

verify:verify.o
	$(CXX) -lz -o verify verify.o
verify.o:verify.C
	$(CXX) $(FLAGS) -o verify.o -c verify.C

clean:
	rm *.o verify
