SRC_DIR=./
MINISAT_DIR=./minisat
OPTS =-g -I. -DUSEPRIORITY -D__ALWAYS_INLINE -DSTATISTIC -DONESOLUTION 

include ../Makefile.common

all: main main_minisat

CMINISATFLAGS = -I$(MINISAT) -DMINISAT

main:  main.o searchEngineAPI.o 
	$(CHARMC) $(OPTS) -language charm++ main.o searchEngineAPI.o -o main -module searchEngine

main.decl.h: $(SRC_DIR)/main.ci
	$(CHARMC) $(OPTS) $(SRC_DIR)/main.ci

main.o: $(SRC_DIR)/defines.h $(SRC_DIR)/main.C main.decl.h
	$(CHARMC) $(OPTS) -c $(SRC_DIR)/main.C

searchEngineAPI.o: $(SRC_DIR)/searchEngineAPI.C $(SRC_DIR)/searchEngineAPI.h
	$(CHARMC) -c  $(SRC_DIR)/searchEngineAPI.C

main_minisat:  main.o searchEngineAPI_minisat.o Solver.o solvers_convertor.o 
	$(CHARMC) $(OPTS) -language charm++ main.o searchEngineAPI_minisat.o Solver.o solvers_convertor.o -o main_minisat -module searchEngine

searchEngineAPI_minisat.o: $(SRC_DIR)/searchEngineAPI.C $(SRC_DIR)/searchEngineAPI.h
	$(CHARMC) $(CMINISATFLAGS) -o searchEngineAPI_minisat.o -c  $(SRC_DIR)/searchEngineAPI.C

Solver.o: $(MINISAT_DIR)/Solver.C $(MINISAT_DIR)/Solver.h  $(MINISAT_DIR)/SolverTypes.h 
	$(CHARMC) $(CMINISATFLAGS)  -c $(MINISAT_DIR)/Solver.C

solvers_convertor.o: $(SRC_DIR)/solvers_convertor.C
	$(CHARMC) $(CMINISATFLAGS) -c $(SRC_DIR)/solvers_convertor.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o main main_minisat charmrun *~
