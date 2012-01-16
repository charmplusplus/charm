#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <zlib.h>
#include <vector>
#include <map>
#include <list>

#include "defines.h"
#include "searchEngine.h"
#include "searchEngineAPI.h"
/*   framework for search engine */

#ifdef MINISAT
#include "solvers_convertor.h"
#endif

using namespace std;

#define CHUNK_LIMIT 1048576
#define MAX_LITS 3
extern int initial_grainsize;
extern char inputfile[];

void SatStateBase::initialize(int vars, int cl, int l)
{
    var_size    =   vars;
    clause_size =   cl;
    level       =   l;
    current_pointer =   0;
    int offset = 0;
    offset = sizeof(SatStateBase);
    clauses = (int*)((char*)this + offset);
    offset += 3*clause_size*sizeof(int);
    occurrences =(int*)( (char*)this + offset);
    offset += var_size *  sizeof(int);
    positive_occurrences = (int*)((char*)this + offset);
    for(int __i=0; __i<var_size; __i++)
    {
        occurrences[__i] = 0;
        positive_occurrences[__i] = 0;
    }
}


int SatStateBase::unsolvedClauses() const
{
    int unsolved = 0;
    for(int i=0; i< clause_size; i++)
    {
        if(clauses[i*MAX_LITS]!= 0)
            unsolved++;
    }
    return unsolved;
}

/* add clause, before adding, check unit conflict */
bool SatStateBase::addClause(int *ps)
{
    /*TODO precheck is needed here */
    for(int i=0; i<MAX_LITS; i++)
    {
        clauses[MAX_LITS*current_pointer + i] = ps[i];
    }
    current_pointer++;
    return true;
}


/**** branching strategy */
int SatStateBase::makeDecision() const
{
    int max_index = -1;
    int max = 0;
    for(int __i=0; __i<var_size; __i++)
    {
        if(occurrences[__i] > max)
        {
            max = occurrences[__i];
            max_index = __i;
        }
    }
    return max_index;

}

inline void SatStateBase::copy(SatStateBase* org)
{
    for(int _i=0; _i<org->clause_size * 3; _i++)
    {
        clauses[_i] = org->clauses[_i];
    }

    for(int _i=0; _i<org->var_size; _i++){
        occurrences[_i] = org->occurrences[_i];
        positive_occurrences[_i] = org->positive_occurrences[_i];
    }
}
inline void SatStateBase::copyto(SatStateBase* dest) const
{
    for(int _i=0; _i<clause_size * 3; _i++)
    {
        (dest->clauses)[_i] = clauses[_i];
    }

    for(int _i=0; _i<var_size; _i++){
        (dest->occurrences)[_i] = occurrences[_i];
        (dest->positive_occurrences)[_i] = positive_occurrences[_i];
    }
}
void SatStateBase::printSolution() const
{
  // This is silly! We shouldn't be using CkError to print the solution,
  // we should be using CkPrintf for this and using CkError for all the other
  // waypoint diagnostic messages. Still, since we have done it backwards for
  // all this time, we might as well continue doing it backwards.
    CkError("One solution:\n");
    for(int _i=0; _i<var_size; _i++)
    {
        CkError("(%d=%s)", _i+1, (occurrences[_i]==-1)?"false":"true");
    }
    CkError("\n");
}

void SatStateBase::printInfo() const
{

    CkPrintf("\n++++++++");
    for(int i=0; i<clause_size; i++)
    {
        CkPrintf("\n");
        for(int j=0; j<MAX_LITS; j++)
        {
            CkPrintf(" %d", clauses[i*MAX_LITS+j]); 
        }
    }
    CkPrintf("occurence:");
    for(int i=0; i<var_size; i++)
    {
        CkPrintf("(%d, %d)", i+1, occurrences[i]);
    }
        CkPrintf("\n");
}
void SatStateBase::printState() const
{ 
    CkPrintf("\n#######State of chare:%d, %d\n", assigned_lit, occurrences[assigned_lit]);
    CkPrintf("level=%d var_size=%d, clauses size=%d\n", level, var_size, clause_size);
    CkPrintf("unsolved clauses  by functions=%d\n",  unsolvedClauses());
    for(int i=0; i<clause_size*MAX_LITS ; i++)
    {
        CkPrintf("%d ",  clauses[i]);
    }
    CkPrintf("\n");

    for(int i=0; i<var_size; i++)
    {
        if(occurrences[i]!=0)
            CkPrintf("(Lit%d,%d)", i+1, occurrences[i]);
    }
    CkPrintf("\n");
}

static void createInitialChildren(Solver *solver)
{

    FILE *file;
    char line[200];
    int vars;
    int clauses;
    char pstring[10], cnfstring[10];
    file = fopen(inputfile, "r");
    if(file == NULL)
    {
        CkPrintf("File not exist\n");
        CkExit();
    }
    /* parse the header and get the number of clauses and variables */
    while(fgets(line, 200, file) != NULL)
    {
        if(strncmp(line, "p cnf", 5) == 0)
        {
            sscanf(line, "%s %s %d %d", pstring, cnfstring, &vars, &clauses);
            break;
        }
    }
    CkPrintf("\n====================================\n");
    CkPrintf("Clauses number:%d, variable number:%d\n\n", clauses, vars);
    /* fill the data with clauses */ 
    SatStateBase *root_1 = (SatStateBase*)solver->registerRootState(sizeof(SatStateBase) + 3*clauses*sizeof(int) + 2*vars*sizeof(int), 0, 2);
    root_1->initialize(vars, clauses, 0);

    int lit, abs_lit;
    int lit_[3];
    int cur_pointer = 0;
    for(int i=0; i<clauses; i++)
    {
        fgets(line, 200, file);
        sscanf(line, "%d %d %d", lit_, lit_+1, lit_+2);
        for(int j=0; j<MAX_LITS; j++)
        {
            lit = lit_[j];
            abs_lit = abs(lit)-1;
            root_1->occurrence(abs_lit)++;
            if(lit>0)
                root_1->positiveOccurrence(lit)++;
            root_1->clause(cur_pointer) = lit;
            cur_pointer++;
        } 
    }
    //root_1->printInfo();
    /* positive assignment */
    int decision_index = root_1->makeDecision();
    root_1->assignedLit() = decision_index+1;
    root_1->occurrence(decision_index) = -2;
    /*negative assignment */
    SatStateBase *root_2 = (SatStateBase*)solver->registerRootState(sizeof(SatStateBase) + 3*clauses*sizeof(int) + 2*vars*sizeof(int), 1, 2);
    root_2->initialize(vars, clauses, 0);
    root_2->copy(root_1);
    root_2->assignedLit() = -decision_index-1;
    root_2->occurrence(decision_index) = -1;
    solver->process(root_1);
    solver->process(root_2);
}

static void createChildren( StateBase *_base , Solver* solver, bool parallel)
{

#ifdef MINISAT
    convertToSequential(_base);
    return;
#endif
    int lit = ((SatStateBase*)_base)->assignedLit();
    int level =((SatStateBase*)_base)->getLevel();
    int clause = ((SatStateBase*)_base)->getClauseSize();
    int vars = ((SatStateBase*)_base)->getVarSize();
    int state_size = sizeof(SatStateBase) + 3*clause*sizeof(int) + 2*vars* sizeof(int);

    SatStateBase* base = (SatStateBase*)alloca(state_size);
    
    base->initialize(vars, clause, 0);
    base->copy((SatStateBase*)_base);


    map<int, list<int> >::iterator pmap;
    //CkPrintf("\nassigned lit=%d\n", lit);
    //base->printInfo();
    SatStateBase *next_state = (SatStateBase*)solver->registerRootState(sizeof(SatStateBase) + 3*clause*sizeof(int) + 2*vars*sizeof(int), 0, 2);
    next_state->initialize(vars, clause, 0);
    next_state->copy(base);

    int vars_eliminate = 0;
    int clause_eliminate = 0;
   
    map<int, list<int> > litMaptoClauses;
    litMaptoClauses.clear();
    for(int i=0; i<next_state->getClauseSize(); i++)
    {
        for(int j=0; j<MAX_LITS &&next_state->clause(i*MAX_LITS+j) !=0; j++)
        {
                int lit_index = next_state->clause(i*MAX_LITS + j);
                litMaptoClauses[lit_index]. push_back(i);
        }
    }
#if 0

    map<int, list<int> >::iterator pmap;
    for(pmap = litMaptoClauses.begin(); pmap != litMaptoClauses.end(); ++pmap)
    {
        list<int> &cl_refer = pmap->second;
        CkPrintf("\n literal %d in clauses:", pmap->first);

        list<int>::iterator iter;
        for(iter=cl_refer.begin(); iter != cl_refer.end(); iter++)
        {
            CkPrintf("%d  ", *iter);
        }
    }


#endif

    int _unit_ = -1;
    vector<int> unit_clause_index;
    /* Unit propagation */
    while(1){
        int pp_ = 1;
        int pp_i_ = 2;
        int pp_j_ = 1;

        if(lit < 0)
        {
            pp_ = -1;
            pp_i_ = 1;
            pp_j_ = 2;
        }
        list<int> &inClauses = litMaptoClauses[lit];
        list<int> &inClauses_opposite = litMaptoClauses[-lit];

        /* literal with same sign, remove all these clauses */
        list<int>::iterator iter;
        for( iter=inClauses.begin(); iter!= inClauses.end(); iter++)
        {
            int cl_index = *iter;
            int begin = cl_index * MAX_LITS;
            int end  = cl_index * MAX_LITS + MAX_LITS;
            /*for all the literals in this clauses, the occurrence decreases by 1 */
            for(int k=begin; k< end&&next_state->clause(k)!=0; k++)
            {
                int  lit_ = next_state->clause(k);
                next_state->occurrence(abs(lit_) - 1)--;
                if(lit_ > 0)
                {
                    next_state->positiveOccurrence(lit_-1)--;
                }
            } //finish dealing with all literal in the clause
            next_state->clause(begin) = 0;
        } //finish dealing with clauses where the literal occur the same
        /* opposite to the literal */
        for(iter= inClauses_opposite.begin(); iter!=inClauses_opposite.end(); iter++)
        {
            int cl_index_ = *iter;
            int mm;
            int begin = cl_index_ *MAX_LITS;
            int end = cl_index_*MAX_LITS + MAX_LITS;

            if(next_state->clause(begin) == 0)
                continue;
            for( mm=begin; mm<end && next_state->clause(mm)!=0;mm++)
            {
                if(next_state->clause(mm) == -lit)
                    break;
            }

            for(;mm<end-1; mm++)
            {
                next_state->clause(mm) = next_state->clause(mm+1);
            }
            next_state->clause(end-1) = 0;
            /*becomes a unit clause */
            if(next_state->clause(begin) == 0)
            { /* conflict */
                //CkPrintf("conflict detected, with assigned lit=%d\n", lit); 
                solver->deleteState(next_state);
                return;
            }else if (next_state->clause(begin+1) == 0)
            {
                unit_clause_index.push_back(cl_index_);
            }
        }
        _unit_++;
        next_state->occurrence(pp_*lit-1) = -pp_i_;
        if(_unit_ == unit_clause_index.size())
            break;
        int cl = next_state->clause(MAX_LITS * unit_clause_index[_unit_]);

        while(cl == 0){
            _unit_++;
            if(_unit_ == unit_clause_index.size())
                break;
            cl = next_state->clause(MAX_LITS * unit_clause_index[_unit_]);
        };

        if(_unit_ == unit_clause_index.size())
            break;
        lit = cl;
    }
    /***************/
    int unsolved = next_state->unsolvedClauses(); 
    if(unsolved == 0)
    {
        //next_state->printInfo();
#ifdef DEBUG
        next_state->printSolution();
#endif
        solver->deleteState(next_state);
        solver->reportSolution();
        return;
    }
    int max_index = next_state->makeDecision();
    SatStateBase *new_msg2 = (SatStateBase*)solver->registerRootState(sizeof(SatStateBase) + 3*clause*sizeof(int) + 2*vars*sizeof(int), 1, 2);
    new_msg2->initialize(vars, clause, 0);
    new_msg2->copy(next_state);

    next_state->assignedLit() = max_index+1;
    next_state->occurrence(max_index) = -2;
    new_msg2->assignedLit() = -max_index-1;
    new_msg2->occurrence(max_index) = -1;

    if(parallel)
    {
        solver->process(next_state);
        solver->process(new_msg2);
    }
}

static bool isGoal(StateBase *s){
    return true;
}

static bool terminate(StateBase *s){
    return true;
}

__INLINE int parallelLevel()
{
        return initial_grainsize;
}

__INLINE int searchDepthLimit()
{
	return 2;
}

SE_Register(SatStateBase, createInitialChildren, createChildren, parallelLevel, searchDepthLimit);
