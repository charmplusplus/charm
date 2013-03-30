MINISAT		  = minisat
TNM			  = TNM
#CHARMDIR = /Users/yanhuasun/PPL/charm/charm-6.1.2/multicore-darwin-x86_64
OPTS = -g
CXX = ../../../bin/charmc

EXES=parsat_minisat parsat_tnm

default: all
all: $(EXES)

CFLAGS    = $(OPTS)  -ffloat-store
CMINISATFLAGS = $(CFLAGS) -I$(MINISAT) -DMINISAT
CTNMFLAGS 	 = $(CFLAGS) -I$(TNM) -DTNM
LFLAGS    =  -lz -language charm++
COPTIMIZE = -O3

CXX_COMPILE = $(CXX)

CXX_LD = $(CXX) $(LFLAGS)

parsat_minisat: $(MINISAT)/Solver.o par_Solver_minisat.o main.o
	$(CXX_LD) $(LFLAGS) -o parsat_minisat $(MINISAT)/Solver.o par_Solver_minisat.o main.o


main.o : main.C main.decl.h main.def.h 
	$(CXX_COMPILE) $(CMINISATFLAGS) -o main.o -c main.C

par_Solver_minisat.o : par_Solver.C par_SolverTypes.h par_Solver.h $(MINISAT)/SolverTypes.h  main.decl.h main.def.h 
	$(CXX_COMPILE) $(CMINISATFLAGS) -o par_Solver_minisat.o -c par_Solver.C

main.def.h main.decl.h : main.ci.stamp

main.ci.stamp: main.ci
	$(CXX) $<
	touch $@

$(MINISAT)/Solver.o: $(MINISAT)/Solver.C $(MINISAT)/Solver.h  $(MINISAT)/SolverTypes.h 
	$(CXX_COMPILE) $(CMINISATFLAGS) -o $(MINISAT)/Solver.o -c $(MINISAT)/Solver.C 


parsat_tnm: $(TNM)/TNM.o par_Solver_tnm.o main.o
	$(CXX) $(LFLAGS) -o parsat_tnm $(TNM)/TNM.o par_Solver_tnm.o main.o 

par_Solver_tnm.o : par_Solver.C par_SolverTypes.h par_Solver.h  main.decl.h main.def.h 
	$(CXX_COMPILE) $(CTNMFLAGS) -o par_Solver_tnm.o -c par_Solver.C 

$(TNM)/TNM.o: $(TNM)/TNM.C
	$(CXX_COMPILE) $(CTNMFLAGS) -o $(TNM)/TNM.o -c $(TNM)/TNM.C

#$(TNM)/adaptnoisebis1at5.o: $(TNM)/adaptnoisebis1at5.C
#	$(CXX_COMPILE) $(CTNMFLAGS) -o $(TNM)/adaptnoisebis1at5.o -c $(TNM)/adaptnoisebis1at5.C

#$(TNM)/inputbis.o : $(TNM)/inputbis.C
#	$(CXX_COMPILE) $(CTNMFLAGS) -o $(TNM)/inputbis.o -c $(TNM)/inputbis.C

clean:
	@rm -f $(EXES) *.def.h *.decl.h *.stamp *.o $(MINISAT)/*.o $(TNM)/*.o charmrun
