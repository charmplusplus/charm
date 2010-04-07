#include "main.decl.h"
#include "par_SolverTypes.h"
#include "par_Solver.h"
#include <map>
#include <vector>

#ifdef MINISAT
#include "Solver.h"
#endif

#ifdef TNM
#include "TNM.h"
#endif

using namespace std;

extern CProxy_Main mainProxy;
extern int grainsize;

par_SolverState::par_SolverState()
{
    var_frequency = 0;
}


int par_SolverState::clausesSize()
{
    return clauses.size();
}

void par_SolverState::attachClause(par_Clause& c)
{

}

int par_SolverState::unsolvedClauses()
{
    int unsolved = 0;
    for(int i=0; i< clauses.size(); i++)
    {
        if(clauses[i].size() > 0)
            unsolved++;
    }

    return unsolved;
}

void par_SolverState::printSolution()
{
    for(int _i=0; _i<var_size; _i++)
    {
        CkPrintf("%d\n", occurrence[_i]);
    }
 
}

/* add clause, before adding, check unit conflict */
bool par_SolverState::addClause(CkVec<par_Lit>& ps)
{
    /*TODO precheck is needed here */
    if(ps.size() == 1)//unit clause
    {
        for(int _i=0; _i<unit_clause_index.size(); _i++)
        {
            int index = unit_clause_index[_i];

            par_Clause unit = clauses[index];
            par_Lit     unit_lit = unit[0];

            if(unit_lit == ~ps[0])
            {
   
                CkPrintf("clause conflict between %d and %d\n", index, clauses.size());
                return false;
            }
        }
       /* all unit clauses are checked, no repeat, no conflict */
        unit_clause_index.push_back(clauses.size());
    }else{
    //check whether the clause is already satisfiable in any case
    /* if x and ~x exist in the same clause, this clause is already satisfiable, we do not have to deal with it */
   
    for(int i=0; i< ps.size(); i++)
    {
        par_Lit lit = ps[i];

        for(int j=0; j<i; j++)
        {
            if(lit == ~ps[j])
            {
                CkPrintf("This clause is already satisfiable\n");
                for(int _i=0; _i<ps.size(); _i++)
                {
                   occurrence[abs(toInt(ps[_i])-1)]--; 
                   if(toInt(ps[_i]) > 0)
                       positive_occurrence[abs(toInt(ps[_i])-1)]--; 
                }
                return true;
            }
        }
    }
    }
    clauses.push_back(par_Clause());
    clauses[clauses.size()-1].attachdata(ps, false);
    // build the linklist for each literal pointing to the clause, where the literal occurs
    for(int i=0; i< ps.size(); i++)
    {
        par_Lit lit = ps[i];

        if(toInt(lit) > 0)
            whichClauses[2*toInt(lit)-2].insert(pair<int, int>(clauses.size()-1, 1));
        else
            whichClauses[-2*toInt(lit)-1].insert(pair<int, int> (clauses.size()-1, 1));

    }

    return true;
}

void par_SolverState::assignclause(CkVec<par_Clause>& cls )
{
    clauses.removeAll();
    for(int i=0; i<cls.size(); i++)
    {
        clauses.push_back( cls[i]);
    }
}

/* *********** Solver chare */

mySolver::mySolver(par_SolverState* state_msg)
{

    /* Which variable get assigned  */
    par_Lit lit = state_msg->assigned_lit;
#ifdef DEBUG    
    CkPrintf("\n\nNew chare: literal = %d, occurrence size=%d, level=%d \n", toInt(lit), state_msg->occurrence.size(), state_msg->level);
#endif    
    par_SolverState *next_state = copy_solverstate(state_msg);

    //Unit clauses
    /* use this value to propagate the clauses */
    // deal with the clauses where this literal is
    int _unit_ = -1;
    while(1){
        int pp_ = 1;
        int pp_i_ = 2;
        int pp_j_ = 1;

        if(toInt(lit) < 0)
        {
            pp_ = -1;
            pp_i_ = 1;
            pp_j_ = 2;
        }

        next_state->occurrence[pp_*toInt(lit)-1] = -pp_i_;
    

        map_int_int &inClauses = next_state->whichClauses[pp_*2*toInt(lit)-pp_i_];
        map_int_int  &inClauses_opposite = next_state->whichClauses[pp_*2*toInt(lit)-pp_j_];

    // literal with same sign, remove all these clauses
       for( map_int_int::iterator iter = inClauses.begin(); iter!=inClauses.end(); iter++)
       {
           int cl_index = (*iter).first;

           par_Clause& cl_ = next_state->clauses[cl_index];
           //for all the literals in this clauses, the occurrence decreases by 1
           for(int k=0; k< cl_.size(); k++)
           {
               par_Lit lit_ = cl_[k];
               if(toInt(lit_) == toInt(lit))
                   continue;
               next_state->occurrence[abs(toInt(lit_)) - 1]--;
               if(toInt(lit_) > 0)
               {
                   next_state->positive_occurrence[toInt(lit_)-1]--;
                   map_int_int::iterator one_it = next_state->whichClauses[2*toInt(lit_)-2].find(cl_index);
                   next_state->whichClauses[2*toInt(lit_)-2].erase(one_it);
               }else
               {
                   map_int_int::iterator one_it = next_state->whichClauses[-2*toInt(lit_)-1].find(cl_index);
                   next_state->whichClauses[-2*toInt(lit_)-1].erase(one_it);
               }
               
           } //finish dealing with all literal in the clause
           next_state->clauses[cl_index].resize(0);
       } //finish dealing with clauses where the literal occur the same
       /* opposite to the literal */
       for(map_int_int::iterator iter= inClauses_opposite.begin(); iter!=inClauses_opposite.end(); iter++)
       {
           int cl_index_ = (*iter).first;
           par_Clause& cl_neg = next_state->clauses[cl_index_];
           cl_neg.remove(-toInt(lit));

           /*becomes a unit clause */
           if(cl_neg.size() == 1)
           {
               next_state->unit_clause_index.push_back(cl_index_);
           }else if (cl_neg.size() == 0)
           {
               return;
           }
       }

       _unit_++;
       if(_unit_ == next_state->unit_clause_index.size())
           break;
       par_Clause cl = next_state->clauses[next_state->unit_clause_index[_unit_]];

       while(cl.size() == 0){
           _unit_++;
           if(_unit_ == next_state->unit_clause_index.size())
               break;
           cl = next_state->clauses[next_state->unit_clause_index[_unit_]];
       };

       if(_unit_ == next_state->unit_clause_index.size())
           break;
       lit = cl[0];


    }
    /***************/

    int unsolved = next_state->unsolvedClauses();
    if(unsolved == 0)
    {
        CkPrintf("One solution found in parallel processing \n");
        //next_state->printSolution();
        mainProxy.done(next_state->occurrence);
        return;
    }
    int max_index = get_max_element(next_state->occurrence);
#ifdef DEBUG
    CkPrintf("max index = %d\n", max_index);
#endif
    //if() left literal unassigned is larger than a grain size, parallel 
    ///* When we start sequential 3SAT Grainsize problem*/
   
    /* the other branch */
    par_SolverState *new_msg2 = copy_solverstate(next_state);;

    next_state->level = state_msg->level+1;

    int lower = state_msg->lower;
    int higher = state_msg->higher;
    int middle = (lower+higher)/2;
    int positive_max = next_state->positive_occurrence[max_index];
    if(positive_max >= next_state->occurrence[max_index] - positive_max)
    {
        next_state->assigned_lit = par_Lit(max_index+1);
        next_state->occurrence[max_index] = -2;
    }
    else
    {
        next_state->assigned_lit = par_Lit(-max_index-1);
        next_state->occurrence[max_index] = -1;
    }
    bool satisfiable_1 = true;

    if(unsolved > grainsize)
    {
        next_state->lower = lower + 1;
        next_state->higher = middle;
        *((int *)CkPriorityPtr(next_state)) = lower+1;
        CkSetQueueing(next_state, CK_QUEUEING_IFIFO);
        CProxy_mySolver::ckNew(next_state);
    }
    else //sequential
    {
        /* This code is urgly. Need to revise it later. Convert par data structure to sequential 
         */
        vector< vector<int> > seq_clauses;
        //seq_clauses.resize(next_state->clauses.size());
        for(int _i_=0; _i_<next_state->clauses.size(); _i_++)
        {
            if(next_state->clauses[_i_].size() > 0)
            {
                vector<int> unsolvedclaus;
                par_Clause& cl = next_state->clauses[_i_];
                unsolvedclaus.resize(cl.size());
                for(int _j_=0; _j_<cl.size(); _j_++)
                {
                    unsolvedclaus[_j_] = toInt(cl[_j_]);
                }
                seq_clauses.push_back(unsolvedclaus);
            }
        }
        satisfiable_1 = seq_processing(next_state->var_size, seq_clauses);//seq_solve(next_state);
        if(satisfiable_1)
        {
            CkPrintf("One solution found by sequential processing \n");
            mainProxy.done(next_state->occurrence);
            return;
        }
    }

    new_msg2->level = state_msg->level+1;
    if(positive_max >= new_msg2->occurrence[max_index] - positive_max)
    {
        new_msg2->assigned_lit = par_Lit(-max_index-1);
        new_msg2->occurrence[max_index] = -1;
    }
    else
    {
        new_msg2->assigned_lit = par_Lit(max_index+1);
        new_msg2->occurrence[max_index] = -2;
    }
    unsolved = new_msg2->unsolvedClauses();
    if(unsolved > grainsize)
    {
        new_msg2->lower = middle + 1;
        new_msg2->higher = higher-1;
        *((int *)CkPriorityPtr(new_msg2)) = middle+1;
        CkSetQueueing(new_msg2, CK_QUEUEING_IFIFO);
        CProxy_mySolver::ckNew(new_msg2);
    }
    else
    {
       
        vector< vector<int> > seq_clauses;
        for(int _i_=0; _i_<new_msg2->clauses.size(); _i_++)
        {
            par_Clause& cl = new_msg2->clauses[_i_];
            if(cl.size() > 0)
            {
                vector<int> unsolvedclaus;
                unsolvedclaus.resize(cl.size());
                for(int _j_=0; _j_<cl.size(); _j_++)
                {
                    unsolvedclaus[_j_] = toInt(cl[_j_]);
                }
                seq_clauses.push_back(unsolvedclaus);
            }
        }

        bool ret = seq_processing(new_msg2->var_size,  seq_clauses);//seq_solve(next_state);
        //bool ret = Solver::seq_processing(new_msg2->clauses);//seq_solve(new_msg2);
        if(ret)
        {
            CkPrintf("One solution found by sequential processing \n");
            mainProxy.done(new_msg2->occurrence);
        }
        return;
    }

}

/* Which literals are already assigned, which is assigned this interation, the unsolved clauses */
/* should all these be passed as function parameters */
/* solve the 3sat in sequence */

long long int computes = 0;
bool mySolver::seq_solve(par_SolverState* state_msg)
{
    /* Which variable get assigned  */
    par_Lit lit = state_msg->assigned_lit;
       
#ifdef DEBUG
    CkPrintf("\n\n Computes=%d Sequential SAT New chare: literal = %d,  level=%d, unsolved clauses=%d\n", computes++, toInt(assigned_var), state_msg->level, state_msg->clauses.size());
    //CkPrintf("\n\n Computes=%d Sequential SAT New chare: literal = %d, occurrence size=%d, level=%d \n", computes++, toInt(assigned_var), state_msg->occurrence.size(), state_msg->level);
#endif
    par_SolverState *next_state = copy_solverstate(state_msg);
    
    //Unit clauses
    /* use this value to propagate the clauses */
#ifdef DEBUG
    CkPrintf(" remainning clause size is %d\n", (state_msg->clauses).size());
#endif

    int _unit_ = -1;
    while(1){
    int pp_ = 1;
    int pp_i_ = 2;
    int pp_j_ = 1;

    if(toInt(lit) < 0)
    {
        pp_ = -1;
        pp_i_ = 1;
        pp_j_ = 2;
    }

    next_state->occurrence[pp_*toInt(lit)-1] = -pp_i_;
    
    map_int_int &inClauses = next_state->whichClauses[pp_*2*toInt(lit)-pp_i_];
    map_int_int &inClauses_opposite = next_state->whichClauses[pp_*2*toInt(lit)-pp_j_];

    // literal with same sign, remove all these clauses
    
    for( map_int_int::iterator iter = inClauses.begin(); iter!=inClauses.end(); iter++)
    {
        int cl_index = (*iter).first;
        par_Clause& cl_ = next_state->clauses[cl_index];
        //for all the literals in this clauses, the occurrence decreases by 1
        for(int k=0; k< cl_.size(); k++)
        {
            par_Lit lit_ = cl_[k];
            if(toInt(lit_) == toInt(lit))
                continue;
            next_state->occurrence[abs(toInt(lit_)) - 1]--;
            if(toInt(lit_) > 0)
            {
                next_state->positive_occurrence[toInt(lit_)-1]--;
                map_int_int::iterator one_it = next_state->whichClauses[2*toInt(lit_)-2].find(cl_index);
                next_state->whichClauses[2*toInt(lit_)-2].erase(one_it);
            }else
            {
                map_int_int::iterator one_it = next_state->whichClauses[-2*toInt(lit_)-1].find(cl_index);
                next_state->whichClauses[-2*toInt(lit_)-1].erase(one_it);
            }

        }
        next_state->clauses[cl_index].resize(0);
    }
   
    for(map_int_int::iterator iter= inClauses_opposite.begin(); iter!=inClauses_opposite.end(); iter++)
    {
        int cl_index_ = (*iter).first;
        par_Clause& cl_neg = next_state->clauses[cl_index_];
        cl_neg.remove(-toInt(lit));
            //becomes a unit clause
         if(cl_neg.size() == 1)
         {
             next_state->unit_clause_index.push_back(cl_index_);
         }else if (cl_neg.size() == 0)
         {
                return false;
         }
    }
   
    _unit_++;
    if(_unit_ == next_state->unit_clause_index.size())
        break;
    par_Clause cl = next_state->clauses[next_state->unit_clause_index[_unit_]];
    
    while(cl.size() == 0){
        _unit_++;
        if(_unit_ == next_state->unit_clause_index.size())
            break;
        cl = next_state->clauses[next_state->unit_clause_index[_unit_]];
        
    };

    if(_unit_ == next_state->unit_clause_index.size())
        break;
    
    lit = cl[0];
    }
   
    int unsolved = next_state->unsolvedClauses();
    if(unsolved == 0)
    {
        CkPrintf("One solution found in sequential processing, check the output file for assignment\n");
        mainProxy.done(next_state->occurrence);
        return true;
    }
    
    /**********************/
    
        /* it would be better to insert the unit literal in order of their occurrence */
        /* make a decision and then fire new tasks */
        /* if there is unit clause, should choose these first??? TODO */
        /* TODO which variable to pick up */
        /*unit clause literal and also which occurrs most times */
        int max_index =  get_max_element(next_state->occurrence);
#ifdef DEBUG
        CkPrintf("max index = %d\n", max_index);
#endif
        next_state->level = state_msg->level+1;

        par_SolverState *new_msg2 = copy_solverstate(next_state);;
        
        int positive_max = next_state->positive_occurrence[max_index];
        if(positive_max >= next_state->occurrence[max_index] - positive_max)
        {
            next_state->occurrence[max_index] = -2;
            next_state->assigned_lit = par_Lit(max_index+1);
        }
        else
        {
            next_state->occurrence[max_index] = -1;
            next_state->assigned_lit = par_Lit(-max_index-1);
        } 

        bool   satisfiable_1 = seq_solve(next_state);
        if(satisfiable_1)
        {
            return true;
        }
        
        new_msg2->level = state_msg->level+1;
       
        if(positive_max >= next_state->occurrence[max_index] - positive_max)
        {
            new_msg2->occurrence[max_index] = -1;
            new_msg2->assigned_lit = par_Lit(-max_index-1);
        }
        else
        {
            new_msg2->occurrence[max_index] = -2;
            new_msg2->assigned_lit = par_Lit(max_index+1);
        } 
            
        bool satisfiable_0 = seq_solve(new_msg2);
        if(satisfiable_0)
        {
            return true;
        }

        //CkPrintf("Unsatisfiable through sequential\n");
        return false;

}
