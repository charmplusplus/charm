#ifndef _PAR_SOLVER_H
#define _PAR_SOLVER_H

#include <map>

using namespace std;


typedef map<int, int> map_int_int;

class par_SolverState : public CMessage_par_SolverState {

private:

public:

    
    par_SolverState();

    int clausesSize();
    
    
    bool addClause(CkVec<par_Lit>& lits);

    void attachClause(par_Clause& c);

    void assignclause(CkVec<par_Clause>& );

    int  unsolvedClauses();
    
    void printSolution();

    CkVec<par_Clause>   clauses;
    par_Lit             assigned_lit;
    int             var_size;
    int             unsolved_clauses;

    int             var_frequency; // 1, -1, 1 freq = 3
    CkVec<int>      unit_clause_index;
    // -2 means true, -1 means false, 0 means anything, > 0 means occurrence
    CkVec<int>      occurrence; 
    CkVec<int>      positive_occurrence; 
    //2 * N (positive, negative)
    CkVec< map_int_int >   whichClauses;
    int             level;
   
    int             lower;
    int             higher;

    //CkVec<Lit>      lit_state;
    //vector<Lit> assigns;

    /*
    void pup(PUP::er &p) {
    
        p|assigns;
        p|clauses;
        p|assigned_lit;
    }*/

    friend par_SolverState* copy_solverstate( const par_SolverState* org)
    {
       par_SolverState *new_state = new (8*sizeof(int))par_SolverState;
        
       new_state->clauses.resize(org->clauses.size());
       for(int _i=0; _i<org->clauses.size(); _i++)
       {
           new_state->clauses[_i] = par_Clause(org->clauses[_i]); 
       }

       new_state->var_size = org->var_size;
       new_state->occurrence.resize(org->occurrence.size());
       new_state->positive_occurrence.resize(org->occurrence.size());
       for(int _i=0; _i<org->occurrence.size(); _i++){
           new_state->occurrence[_i] = org->occurrence[_i];
           new_state->positive_occurrence[_i] = org->occurrence[_i];
       }
       new_state->level = org->level;
       int _size = org->whichClauses.size();
       new_state->whichClauses.resize(_size); 
       for(int i=0; i<_size; i++)
       {
           map<int, int> __cl_copy = org->whichClauses[i];
           for(map_int_int::iterator iter=__cl_copy.begin(); iter!=__cl_copy.end(); iter++)
           {
               (new_state->whichClauses[i]).insert( pair<int, int>((*iter).first, (*iter).second));
           }
       }
       return new_state;
    }

};


class mySolver : public CBase_mySolver {

private:

    bool seq_solve(par_SolverState*);
public:
    mySolver(par_SolverState*);

};
#endif
