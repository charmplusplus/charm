include Makefile.in
all: multirun

#Charm++ libraries to be used for Interoperation
LIBS=libmodulehi.a libmodulehello.a libmodulekNeighbor.a

multirun: multirun.cpp $(LIBS)
	$(CXX) -c multirun.cpp -o multirun.o -I$(CHARMDIR)/include
	$(CHARMC) -mpi -o multirun multirun.o -L./ -module hello -module hi -module kNeighbor -module CommonLBs 

$(LIBS):  
	cd libs;make;

clean: clear
	cd libs;make clean;cd ..;
	rm -f multirun *.o *.a charmrun

clear:
	rm -f PI*

test: all
	mpirun -np 16 ./multirun
	mpirun -np 16 ./multirun +balancer GreedyLB +LBDebug 1

