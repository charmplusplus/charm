include Makefile.in
all: multirun

#Charm++ libraries to be used for Interoperation
LIBS=libmodulempicollide.a

multirun: multirun.cpp $(LIBS)
	$(CXX) -c multirun.cpp -o multirun.o -I$(CHARMDIR)/include
	$(CHARMC) -mpi -o multirun multirun.o -L./collidelib -module mpicollide -module CommonLBs -module collidecharm

$(LIBS):  
	cd collidelib;make;

clean: clear
	cd collidelib;make clean;cd ..;
	rm -f multirun *.o *.a charmrun

clear:
	rm -f PI*

test: all
	mpirun -np 16 ./multirun
	mpirun -np 16 ./multirun +balancer GreedyLB +LBDebug 1

