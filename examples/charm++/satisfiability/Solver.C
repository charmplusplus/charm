#include "main.decl.h"
#include "SolverTypes.h"
#include "Solver.h"
#include <map>
#include <vector>

using namespace std;

extern CProxy_Main mainProxy;
extern int grainsize;

SolverState::SolverState()
{

    var_frequency = 0;
}

inline int SolverState::nVars()
{
    return var_frequency; 
}

Var SolverState::newVar(bool sign, bool dvar)
{
    int v = nVars();
    var_frequency++;
    return v;
}

void SolverState::attachClause(Clause& c)
{

}

/* add clause, before adding, check unit conflict */
bool SolverState::addClause(CkVec<Lit>& ps)
{
    /*TODO precheck is needed here */
    if(ps.size() == 1)//unit clause
    {
        for(int _i=0; _i<unit_clause_index.size(); _i++)
        {
            int index = unit_clause_index[_i];

            Clause unit = clauses[index];
            Lit     unit_lit = unit[0];

            if(unit_lit == ~ps[0])
            {
   
                CkPrintf("clause conflict between %d and %d\n", index, clauses.size());
                return false;
            }
            /*else if(unit_lit == ps[0])
            {
                return true;
            }*/
        }
       /* all unit clauses are checked, no repeat, no conflict */
        unit_clause_index.push_back(clauses.size());
    }else{
    //check whether the clause is already satisfiable in any case
    /* if x and ~x exist in the same clause, this clause is already satisfiable, we do not have to deal with it */
   
    for(int i=0; i< ps.size(); i++)
    {
        Lit lit = ps[i];

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
    clauses.push_back(Clause());
    clauses[clauses.size()-1].attachdata(ps, false);
    return true;
}

void SolverState::assignclause(CkVec<Clause>& cls )
{
    clauses.removeAll();
    for(int i=0; i<cls.size(); i++)
    {
        clauses.push_back( cls[i]);
    }
}

/* *********** Solver chare */

Solver::Solver(SolverState* state_msg)
{

    CkVec<Clause> next_clauses;
    /* Which variable get assigned  */
    Lit assigned_var = state_msg->assigned_lit;
#ifdef DEBUG    
    CkPrintf("\n\nNew chare: literal = %d, occurrence size=%d, level=%d \n", toInt(assigned_var), state_msg->occurrence.size(), state_msg->level);
#endif    
    SolverState *next_state = copy_solverstate(state_msg);
    
    //Unit clauses
    map<int, int>  unit_clauses;

    /* use this value to propagate the clauses */
    for(int i=0; i<(state_msg->clauses).size(); i++)
    {
        Clause cl = ((state_msg->clauses)[i]);
        CkVec<Lit> new_clause;
        bool satisfied = false;
#ifdef DEBUG    
//        CkPrintf("\n");
#endif
        for(int j=0; j<cl.size();  j++)
        {
            Lit lit = cl[j];
#ifdef DEBUG    
        //    CkPrintf("%d   ", toInt(lit));
#endif
            if(lit == assigned_var)
            {
                satisfied = true;
                break;
            }else if( lit == ~(assigned_var))
            {
            } else
            {
                new_clause.push_back(lit);
            }
        }
        if(satisfied)
        {
            for(int j=0; j<cl.size();  j++)
            {
                Lit lit = cl[j];
                if(lit != assigned_var)
                {
                    int int_lit = toInt(lit);
                    (next_state->occurrence[abs(int_lit)-1])--;
                    if(int_lit > 0)
                        (next_state->positive_occurrence[abs(int_lit)-1])--;
                }
            }
        } else if(new_clause.size() == 0 )
        {
        
            /* conflict */
            return;
        }else if(new_clause.size() == 1)
        {
            /*unit clause */
            Lit lit = new_clause[0];
            if(unit_clauses.find(-toInt(lit)) !=unit_clauses.end())
            {
#ifdef DEBUG
                CkPrintf("conflict detected by unit progation, size=%d\n", unit_clauses.size());
#endif
                return;
                /* conflict*/
            }else
            {
                (unit_clauses[toInt(lit)])++;
                int int_lit = toInt(lit);
                int_lit>0?state_msg->occurrence[int_lit-1]=-2:state_msg->occurrence[-int_lit-1]=-1;
#ifdef DEBUG
                CkPrintf(" unit clause %d=%d, size=%D\n", toInt(lit), unit_clauses[toInt(lit)], unit_clauses.size());
#endif
            }
        }
        else if(new_clause.size() > 1)
        {
            Clause clsc(new_clause, false);
            next_clauses.push_back(clsc);
        }
    } /* all clauses are checked*/
   
    if(next_clauses.size() == 0)
    {
        /* one solution is found*/
        /* print the solution*/
        CkPrintf("one solutions found level=%d parallel\n", state_msg->level);
       /* 
        for(int _i=0; _i<state_msg->var_size; _i++)
        {
            if(state_msg->occurrence[_i] == -2)
                CkPrintf(" TRUE ");
            else if(state_msg->occurrence[_i] == -1)
                CkPrintf(" FALSE ");
            else
                CkPrintf(" Anything ");
        }*/
        CkPrintf("\n");
        mainProxy.done();//highest priority
        //CkExit();
        return;
    }else
    {
        /* make a decision and then fire new tasks */
        /* if there is unit clause, should choose these first??? TODO */
        /* TODO which variable to pick up */
        /*unit clause literal and also which occurrs most times */
        while(!unit_clauses.empty())
        {
           //CkPrintf(" parallel propagation unit clauses size=%d, non unit size=%d, assigned var=%d\n", unit_clauses.size(), next_clauses.size(), toInt(assigned_var));
            int first_unit = (*(unit_clauses.begin())).first;
           //assign this variable
            for(int _i=0; _i<next_clauses.size(); _i++)
            {
                Clause& cl = next_clauses[_i];
               
                //reduce the clause
                for(int j=0; j<cl.size(); j++)
                {
                    Lit lit = cl[j];
                    if(toInt(lit) == first_unit)//this clause goes away
                    {
                        for(int jj=0; jj<cl.size(); jj++)
                        {
                            Lit lit_2 = cl[jj];
                            (next_state->occurrence[abs(toInt(lit_2))-1])--; 
                            if(toInt(lit_2) > 0)
                                (next_state->positive_occurrence[abs(toInt(lit_2))-1])--; 
                        }
                        next_clauses.remove(_i);
                        _i--;
                       break; 
                    }else if (toInt(lit) == -first_unit)
                    {
                        cl.remove(j);
                        j--;
                    }
                }
                if(cl.size()==0){ // conflict
                    return;
                }else if(cl.size() == 1)// new unit clause
                {
                    unit_clauses[toInt(cl[0])]++;
                
                    next_clauses.remove(_i);
                    _i--;
                }
            }
            /*remove this unit clause */
            unit_clauses.erase(unit_clauses.begin());
        }

        }

    if(next_clauses.size() == 0)
    {
        CkPrintf(" one solutions found after propagation \n");
        mainProxy.done();
        //CkExit();
        return;
    }
        int max_index = get_max_element(next_state->occurrence);
#ifdef DEBUG
        CkPrintf("max index = %d\n", max_index);
#endif
        if(next_state->occurrence[max_index] <=0)
        {
            CkPrintf("Unsatisfiable\n");
            mainProxy.done();
            //CkExit();
            return;
        }


        //if() left literal unassigned is larger than a grain size, parallel 
        /* When we start sequential 3SAT Grainsize problem*/
        next_state->assignclause(next_clauses);
        next_state->var_size = state_msg->var_size;

        SolverState *new_msg2 = copy_solverstate(next_state);;
        new_msg2->var_size = state_msg->var_size;

        next_state->level = state_msg->level+1;
       

        int lower = state_msg->lower;
        int higher = state_msg->higher;
        int middle = (lower+higher)/2;
        int positive_max = next_state->positive_occurrence[max_index];
        if(positive_max >= next_state->occurrence[max_index] - positive_max)
        {
            next_state->assigned_lit = Lit(max_index+1);
            next_state->occurrence[max_index] = -2;
        }
        else
        {
            next_state->assigned_lit = Lit(-max_index-1);
            next_state->occurrence[max_index] = -1;
        }
        bool satisfiable_1 = true;
        bool satisfiable_0 = true;

        if(next_state->clauses.size() > grainsize)
        {
            next_state->lower = lower + 1;
            next_state->higher = middle;
            *((int *)CkPriorityPtr(next_state)) = lower+1;
            CkSetQueueing(next_state, CK_QUEUEING_IFIFO);
            CProxy_Solver::ckNew(next_state);
        }
        else //sequential
        {
            satisfiable_1 = seq_solve(next_state);
            if(satisfiable_1)
            {
                CkPrintf(" One solutions found by sequential algorithm\n");
                mainProxy.done();
                //CkExit();
                return;
            }
        }

        new_msg2->level = state_msg->level+1;
        if(positive_max >= new_msg2->occurrence[max_index] - positive_max)
        {
            new_msg2->assigned_lit = Lit(-max_index-1);
            new_msg2->occurrence[max_index] = -1;
        }
        else
        {
            new_msg2->assigned_lit = Lit(max_index+1);
            new_msg2->occurrence[max_index] = -2;
        }
        if(new_msg2->clauses.size() > grainsize)
        {

            new_msg2->lower = middle + 1;
            new_msg2->higher = higher-1;
            *((int *)CkPriorityPtr(new_msg2)) = middle+1;
            CkSetQueueing(new_msg2, CK_QUEUEING_IFIFO);
            CProxy_Solver::ckNew(new_msg2);
        }
        else
        {
            satisfiable_0 = seq_solve(new_msg2);

            if(satisfiable_0)
            {
                CkPrintf(" One solutions found by sequential algorithm\n");
                mainProxy.done();
                //CkExit();
                return;
            }

        }

        if(!satisfiable_1 && !satisfiable_0)
        {
            CkPrintf("Unsatisfiable through sequential\n");
            mainProxy.done();
            //CkExit();
        }
}

/* Which literals are already assigned, which is assigned this interation, the unsolved clauses */
/* should all these be passed as function parameters */
/* solve the 3sat in sequence */

long long int computes = 0;
bool Solver::seq_solve(SolverState* state_msg)
{
       
    CkVec<Clause> next_clauses;
    /* Which variable get assigned  */
    Lit assigned_var = state_msg->assigned_lit;
#ifdef DEBUG
    CkPrintf("\n\n Computes=%d Sequential SAT New chare: literal = %d,  level=%d, unsolved clauses=%d\n", computes++, toInt(assigned_var), state_msg->level, state_msg->clauses.size());
    //CkPrintf("\n\n Computes=%d Sequential SAT New chare: literal = %d, occurrence size=%d, level=%d \n", computes++, toInt(assigned_var), state_msg->occurrence.size(), state_msg->level);
#endif
    SolverState *next_state = copy_solverstate(state_msg);
    
    //Unit clauses
    /* use this value to propagate the clauses */
    map<int, int> unit_clauses;
#ifdef DEBUG
    CkPrintf(" remainning clause size is %d\n", (state_msg->clauses).size());
#endif
    for(int i=0; i<(state_msg->clauses).size(); i++)
    {
        Clause cl = ((state_msg->clauses)[i]);
        CkVec<Lit> new_clause;
        bool satisfied = false;
#ifdef DEBUG    
        //CkPrintf("\n");
#endif
        for(int j=0; j<cl.size();  j++)
        {
            Lit lit = cl[j];
#ifdef DEBUG    
        //    CkPrintf("%d   ", toInt(lit));
#endif
            if(lit == assigned_var)
            {
                satisfied = true;
                break;
            }else if( lit == ~(assigned_var))
            {
            } else
            {
                new_clause.push_back(lit);
            }
        }
        if(satisfied)//one literal is true in the clause
        {           // the occurrence of all other literals in this clause decrease by 1 since this clause goes away
            for(int j=0; j<cl.size();  j++)
            {
                Lit lit = cl[j];
                if(lit != assigned_var)
                {
                    int int_lit = toInt(lit);
                    (next_state->occurrence[abs(int_lit)-1])--;
                    if(int_lit > 0)
                        (next_state->positive_occurrence[abs(int_lit)-1])--;
                }
            }
        } else if(new_clause.size() == 0 ) //this clause is unsatisfiable
        {
        
            /* conflict */
            return false;
        }else if(new_clause.size() == 1) //unit clause
        {
            /*unit clause */
            Lit lit = new_clause[0];
            if(unit_clauses.find(-toInt(lit)) != unit_clauses.end()) // for a variable, x and ~x exist in unit clause at the same time
            {
                CkPrintf("conflict detected by unit progation\n");
                return false;
                /* conflict*/
            }else
            {
                unit_clauses[toInt(lit)]++;
            }

        }
        else if(new_clause.size() > 1)
        {
            Clause clsc(new_clause, false);
            next_clauses.push_back(clsc);
        }
    } /* all clauses are checked*/
   
    //CkPrintf(" After assignment, unit clauses = %d, unsolved clauses=%d\n", unit_clauses.size(), next_clauses.size());

    if(next_clauses.size() == 0)
    {
        /* one solution is found*/
        /* print the solution*/
        //CkPrintf("Inside sequential, done!! one solutions found level=%d by sequential SAT\n", state_msg->level);
        /*
        for(int _i=0; _i<state_msg->var_size; _i++)
        {
            if(state_msg->occurrence[_i] == -2)
                CkPrintf(" TRUE ");
            else if(state_msg->occurrence[_i] == -1)
                CkPrintf(" FALSE ");
            else
                CkPrintf(" Anything ");
        }*/
        CkPrintf("\n");
        return true;
    }else
    {
        /*   assigned all unit clause literals first*/
        while(!unit_clauses.empty())
        {
#ifdef DEBUG
            CkPrintf("unit propagation, unit clauses=%d, unsolved clauses=%d\n", unit_clauses.size(), next_clauses.size());
#endif
            int first_unit = (*(unit_clauses.begin())).first;

            //assign this variable
            for(int _i=0; _i<next_clauses.size(); _i++)
            {
                Clause& cl = next_clauses[_i];
                //reduce the clause
                for(int j=0; j<cl.size(); j++)
                {
                    Lit lit = cl[j];
                    if(toInt(lit) == first_unit)//this clause goes away
                    {
                        for(int jj=0; jj<cl.size(); jj++)
                        {
                            Lit lit_2 = cl[jj];
                            (next_state->occurrence[abs(toInt(lit_2))-1])--; 
                            if(toInt(lit_2) > 0)
                                (next_state->positive_occurrence[abs(toInt(lit_2))-1])--; 
                        }
                       break; 
                    }else if (toInt(lit) == -first_unit)
                    {
                        cl.remove(j);
                        j--;
                    }
                }
                if(cl.size()==0){ // conflict
                    return false;
                    //_i--;
                }else if(cl.size() == 1)// new unit clause
                {
                    unit_clauses[toInt(cl[0])]++;
                    next_clauses.remove(_i);
                    _i--;
                }
            }
            /*remove this unit clause */
            unit_clauses.erase(unit_clauses.begin());
        }

        //CkPrintf("After unit propagation, unit size = %d, unsolved = %d\n", unit_clauses.size(), next_clauses.size());
        if(next_clauses.size() == 0)
            return true;
        /* it would be better to insert the unit literal in order of their occurrence */

        /* make a decision and then fire new tasks */
        /* if there is unit clause, should choose these first??? TODO */
        /* TODO which variable to pick up */
        /*unit clause literal and also which occurrs most times */
        int max_index =  get_max_element(next_state->occurrence);
#ifdef DEBUG
        CkPrintf("max index = %d\n", max_index);
#endif
        if(next_state->occurrence[max_index] <=0)
        {
            CkPrintf("****Unsatisfiable in sequential SAT\n");
            return false;
        }


        next_state->assignclause(next_clauses);
        next_state->var_size = state_msg->var_size;
        next_state->level = state_msg->level+1;

        int positive_max = next_state->positive_occurrence[max_index];
        if(positive_max >= next_state->occurrence[max_index] - positive_max)
        {
            next_state->occurrence[max_index] = -2;
            next_state->assigned_lit = Lit(max_index+1);
        }
        else
        {
            next_state->occurrence[max_index] = -1;
            next_state->assigned_lit = Lit(-max_index-1);
        } 

        bool   satisfiable_1 = seq_solve(next_state);
        if(satisfiable_1)
        {
            return true;
        }
        
       
        if(positive_max >= next_state->occurrence[max_index] - positive_max)
        {
            next_state->occurrence[max_index] = -1;
            next_state->assigned_lit = Lit(-max_index-1);
        }
        else
        {
            next_state->occurrence[max_index] = -2;
            next_state->assigned_lit = Lit(max_index+1);
        } 
            
        bool satisfiable_0 = seq_solve(next_state);
        if(satisfiable_0)
        {
            return true;
        }

        CkPrintf("Unsatisfiable through sequential\n");
        return false;
    }

}
