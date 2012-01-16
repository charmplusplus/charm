#ifdef MINISAT
#include "searchEngineAPI.h"
#include "Solver.h"

inline bool convertToSequential(StateBase *_base)
{
    SatStateBase* parent = (SatStateBase*)_base;
    vector< vector<int> > seq_clauses;
    for(int _i_=0; _i_<parent->clause_size; _i_++)
    {
        if(parent->clauses[_i_*MAX_LITS] != 0)
        {
            vector<int> unsolvedclaus;
            for(int _j_=_i_*MAX_LITS; _j_<_i_*MAX_LITS+MAX_LITS && parent->clauses[_j_]! =0; _j_++)
            {
                unsolvedclaus.push_back(parent->clauses[_j_]);
            }
            seq_clauses.push_back(unsolvedclaus);
        }
    }

    signed int *seq_assignment = (signed int*) malloc( sizeof(signed int) *  parent->getVarSize());
    bool satisfiable = seq_processing(parent->getVarSize(), seq_clauses, seq_assignment);
    return satisfiable;
}

#endif
