MTL       = mtl
EXEC      = par_sat
SEQ		  = core_sequential
#CHARMDIR = /Users/yanhuasun/PPL/charm/charm-6.1.2/multicore-darwin-x86_64
OPTS = -g
CXX = charmc   
default: all
all: $(EXEC)

CFLAGS    = $(OPTS) -I$(MTL)  -I$(SEQ) -ffloat-store
LFLAGS    =  -lz -language charm++
COPTIMIZE = -O3

CXX_COMPILE = $(CXX) $(CFLAGS)

CXX_LD = $(CXX) $(LFLAGS)

$(EXEC): $(SEQ)/Solver.o par_Solver.o main.o
	$(CXX_LD) $(LFLAGS) -o $(EXEC) $(SEQ)/Solver.o par_Solver.o main.o

main.o : main.C main.decl.h main.def.h 
	$(CXX_COMPILE) -o main.o -c main.C

par_Solver.o : par_Solver.C par_SolverTypes.h par_Solver.h mtl/Vec.h mtl/minisatHeap.h  mtl/Alg.h $(SEQ)/SolverTypes.h mtl/Sort.h main.decl.h main.def.h 
	$(CXX_COMPILE) -o par_Solver.o -c par_Solver.C

main.def.h main.decl.h : main.ci
	$(CXX) main.ci

$(SEQ)/Solver.o: $(SEQ)/Solver.C $(SEQ)/Solver.h mtl/Vec.h mtl/minisatHeap.h  mtl/Alg.h $(SEQ)/SolverTypes.h mtl/Sort.h
	$(CXX_COMPILE) -o $(SEQ)/Solver.o -c $(SEQ)/Solver.C 

clean:
	@rm -f $(EXEC) *.def.h *.decl.h *.o $(SEQ)/*.o charmrun
