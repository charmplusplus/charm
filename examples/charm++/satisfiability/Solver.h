#ifndef _SOLVER_H
#define _SOLVER_H

class SolverState : public CMessage_SolverState {

private:

public:

    
    SolverState();

    int nVars();

    Var newVar (bool polarity = true, bool dvar = true);
    
    bool addClause(CkVec<Lit>& lits);

    void attachClause(Clause& c);

    void assignclause(CkVec<Clause>& );


    CkVec<Clause>   clauses;
    Lit             assigned_lit;
    int             var_size;

    int             var_frequency; // 1, -1, 1 freq = 3
    CkVec<int>      unit_clause_index;
    // -2 means true, -1 means false, 0 means anything, > 0 means occurrence
    CkVec<int>      occurrence; 
    CkVec<int>      positive_occurrence; 
    int             level;
    
    //CkVec<Lit>      lit_state;
    //vector<Lit> assigns;

    /*
    void pup(PUP::er &p) {
    
        p|assigns;
        p|clauses;
        p|assigned_lit;
    }*/

    friend SolverState* copy_solverstate( const SolverState* org)
    {
       SolverState *new_state = new SolverState;
        
       new_state->clauses.resize(org->clauses.size());
       for(int _i=0; _i<org->clauses.size(); _i++)
       {
           new_state->clauses[_i] = Clause(org->clauses[_i]); 
       }

       new_state->var_size = org->var_size;

       new_state->occurrence.resize(org->occurrence.size());
       new_state->positive_occurrence.resize(org->occurrence.size());
       for(int _i=0; _i<org->occurrence.size(); _i++){
           new_state->occurrence[_i] = org->occurrence[_i];
           new_state->positive_occurrence[_i] = org->occurrence[_i];
       }
        new_state->level = org->level;
       return new_state;
    }

};


class Solver : public CBase_Solver {

private:

    bool seq_solve(SolverState*);
public:
    Solver(SolverState*);

};
#endif
