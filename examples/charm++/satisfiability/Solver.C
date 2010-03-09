#include "main.decl.h"
#include "SolverTypes.h"
#include "Solver.h"
#include <map>


using namespace std;

extern CProxy_Main mainProxy;

SolverState::SolverState()
{

}

inline int SolverState::nVars()
{
    return assigns.size(); 
}

Var SolverState::newVar(bool sign, bool dvar)
{
    int v = nVars();
    assigns.push_back(toInt(l_Undef));
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
//#ifdef DEBUG    
    CkPrintf("\n\nNew chare: literal = %d, occurrence size=%d, level=%d \n", toInt(assigned_var), state_msg->occurrence.size(), state_msg->level);
//#endif    
    SolverState *next_state = copy_solverstate(state_msg);
    
    //Unit clauses
    map<int, int> unit_clauses;
    //unit_clauses.erase(unit_clauses.begin(), unit_clauses.end());
#ifdef DEBUG
            CkPrintf("map size:%d", unit_clauses.size());
#endif

    /* use this value to propagate the clauses */
    for(int i=0; i<(state_msg->clauses).size(); i++)
    {
        Clause cl = ((state_msg->clauses)[i]);
        CkVec<Lit> new_clause;
        bool satisfied = false;
#ifdef DEBUG    
        CkPrintf("##");
#endif
        for(int j=0; j<cl.size();  j++)
        {
            Lit lit = cl[j];
#ifdef DEBUG    
            CkPrintf("%d   ", toInt(lit));
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
                    (next_state->occurrence[abs(toInt(lit))-1])--;
                }
            }
        } else if(new_clause.size() == 0 )
        {
        
            /* conflict */
            //CkPrintf(" conflict\n");
            //mainProxy.done();
            return;
        }else if(new_clause.size() == 1)
        {
            /*unit clause */
            Lit lit = new_clause[0];
            if(unit_clauses[-toInt(lit)] > 0)
            {
#ifdef DEBUG
                CkPrintf("conflict detected by unit progation\n");
#endif
                return;
                /* conflict*/
            }else
            {
                unit_clauses[toInt(lit)]++;
#ifdef DEBUG
                CkPrintf(" unit clause %d=%d\n", toInt(lit), unit_clauses[toInt(lit)]);
#endif
            }
            Clause clsc(new_clause, false);
            next_clauses.push_back(clsc);
        }
        else if(new_clause.size() > 1)
        {
            Clause clsc(new_clause, false);
            next_clauses.push_back(clsc);
        }
    } /* all clauses are checked*/
   
    //CkPrintf (" new clauses size=%d\n", next_clauses.size());
    if(next_clauses.size() == 0)
    {
        /* one solution is found*/
        /* print the solution*/
        CkPrintf("one solutions found level=%d parallel\n", state_msg->level);
        
        for(int _i=0; _i<state_msg->var_size; _i++)
        {
            if(state_msg->occurrence[_i] == -2)
                CkPrintf(" TRUE ");
            else if(state_msg->occurrence[_i] == -1)
                CkPrintf(" FALSE ");
            else
                CkPrintf(" Anything ");
        }
        CkPrintf("\n");
        mainProxy.done();
        return;
    }else
    {
        /* make a decision and then fire new tasks */
        /* if there is unit clause, should choose these first??? TODO */
        /* TODO which variable to pick up */
        /*unit clause literal and also which occurrs most times */
        int max_index =-1;//= get_max_element(next_state->occurrence);
        if(!unit_clauses.empty())
        {
       
#ifdef DEBUG
            CkPrintf("map size:%d", unit_clauses.size());
#endif
            int max_occur = -3;
            for(int _i=0; _i<next_state->occurrence.size(); _i++)
            {
#ifdef DEBUG
                    CkPrintf(" unit detect index = %d, i=%d, occ=%d, unit_clause=%d\n", max_index, _i, next_state->occurrence[_i], unit_clauses[_i]);
#endif
                if(next_state->occurrence[_i]>max_occur&&(unit_clauses[_i+1]>0 || unit_clauses[-_i-1]>0))
                {
                    max_index = _i;
                    max_occur = next_state->occurrence[_i];
                }
            }
        }else 
            max_index = get_max_element(next_state->occurrence);
#ifdef DEBUG
        CkPrintf("max index = %d\n", max_index);
#endif
        if(next_state->occurrence[max_index] <=0)
        {
            CkPrintf("Unsatisfiable\n");
            mainProxy.done();
            return;
        }


        //if() left literal unassigned is larger than a grain size, parallel 
        /* When we start sequential 3SAT Grainsize problem*/
        next_state->assigned_lit = Lit(max_index+1);
        next_state->assignclause(next_clauses);
        //CkPrintf (" assign clauses size=%d\n", next_state->clauses.size());
        next_state->var_size = state_msg->var_size;

        SolverState *new_msg2 = copy_solverstate(next_state);;
        new_msg2->assigned_lit = Lit(-max_index-1);
        new_msg2->var_size = state_msg->var_size;

        next_state->level = state_msg->level+1;
        next_state->occurrence[max_index] = -2;
        bool satisfiable_1 = true;
        bool satisfiable_0 = true;
        if(unit_clauses.empty()||unit_clauses[max_index+1]>0)        
        {
            if(next_state->clauses.size() > 30)
                CProxy_Solver::ckNew(next_state);
            else //sequential
            {
                satisfiable_1 = seq_solve(next_state);
                if(satisfiable_1)
                {
                    //CkPrintf(" One solutions found by sequential algorithm\n");
                    mainProxy.done();
                    return;
                }
            }
        }
        new_msg2->level = state_msg->level+1;
        new_msg2->occurrence[max_index] = -1;
        if(unit_clauses.empty()||unit_clauses[-max_index-1]>0)
        {
            if(new_msg2->clauses.size() > 30)
                CProxy_Solver::ckNew(new_msg2);
            else
            {
                satisfiable_0 = seq_solve(new_msg2);

                if(satisfiable_0)
                {
                    //CkPrintf(" One solutions found by sequential algorithm\n");
                    mainProxy.done();
                    return;
                }

            }
        }

        if(!satisfiable_1 && !satisfiable_0)
        {
            CkPrintf("Unsatisfiable through sequential\n");
            mainProxy.done();
        }
        // grain size is small now. use sequential to solve it    
        // seq_solve()
    }
}

/* Which literals are already assigned, which is assigned this interation, the unsolved clauses */
/* should all these be passed as function parameters */
/* solve the 3sat in sequence */
bool Solver::seq_solve(SolverState* state_msg)
{
        
    CkVec<Clause> next_clauses;
    /* Which variable get assigned  */
    Lit assigned_var = state_msg->assigned_lit;
    CkPrintf("\n\nSequential SAT New chare: literal = %d, occurrence size=%d, level=%d \n", toInt(assigned_var), state_msg->occurrence.size(), state_msg->level);
    SolverState *next_state = copy_solverstate(state_msg);
    
    //Unit clauses
    map<int, int> unit_clauses;
    //unit_clauses.erase(unit_clauses.begin(), unit_clauses.end());
#ifdef DEBUG
            CkPrintf("map size:%d", unit_clauses.size());
#endif

    /* use this value to propagate the clauses */
    for(int i=0; i<(state_msg->clauses).size(); i++)
    {
        Clause cl = ((state_msg->clauses)[i]);
        CkVec<Lit> new_clause;
        bool satisfied = false;
#ifdef DEBUG    
        CkPrintf("##");
#endif
        for(int j=0; j<cl.size();  j++)
        {
            Lit lit = cl[j];
#ifdef DEBUG    
            CkPrintf("%d   ", toInt(lit));
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
                    (next_state->occurrence[abs(toInt(lit))-1])--;
                }
            }
        } else if(new_clause.size() == 0 ) //this clause is unsatisfiable
        {
        
            /* conflict */
            //CkPrintf(" conflict\n");
            return false;
        }else if(new_clause.size() == 1) //unit clause
        {
            /*unit clause */
            Lit lit = new_clause[0];
            if(unit_clauses[-toInt(lit)] > 0) // for a variable, x and ~x exist in unit clause at the same time
            {
                CkPrintf("conflict detected by unit progation\n");
                return false;
                /* conflict*/
            }else
            {
                unit_clauses[toInt(lit)]++;
#ifdef DEBUG
                CkPrintf(" unit clause %d=%d\n", toInt(lit), unit_clauses[toInt(lit)]);
#endif
            }

            Clause clsc(new_clause, false);
            next_clauses.push_back(clsc);
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
        CkPrintf("one solutions found level=%d by sequential SAT\n", state_msg->level);
        
        for(int _i=0; _i<state_msg->var_size; _i++)
        {
            if(state_msg->occurrence[_i] == -2)
                CkPrintf(" TRUE ");
            else if(state_msg->occurrence[_i] == -1)
                CkPrintf(" FALSE ");
            else
                CkPrintf(" Anything ");
        }
        CkPrintf("\n");
        return true;
    }else
    {
        /*   assigned all unit clause literals first*/
        while(!unit_clauses.empty())
        {
            int last_unit = (*(unit_clauses.begin())).first;

            //assign this variable
            for(int _i=0; _i<next_state->clauses.size(); _i++)
            {
                Clause& cl = (next_state->clauses)[_i];
               
                //reduce the clause
                for(int j=0; j<cl.size(); j++)
                {
                    Lit lit = cl[j];
                    if(toInt(lit) == last_unit)//this clause goes away
                    {
                        for(int jj=0; jj<cl.size(); jj++)
                        {
                            Lit lit_2 = cl[jj];
                            (next_state->occurrence[abs(toInt(lit_2))-1])--; 
                        }
                       break; 
                    }else if (toInt(lit) == -last_unit)
                    {
                        cl.remove(j);
                        j--;
                    }
                }
                if(cl.size()==0){
                    next_state->clauses.remove(_i);
                    _i--;
                }else if(cl.size() == 1)// new unit clause
                {
                    unit_clauses[toInt(cl[0])]++;

                }
            }
            /*remove this unit clause */
            unit_clauses.erase(unit_clauses.begin());
        }

        /* it would be better to insert the unit literal in order of their occurrence */

        /* make a decision and then fire new tasks */
        /* if there is unit clause, should choose these first??? TODO */
        /* TODO which variable to pick up */
        /*unit clause literal and also which occurrs most times */
        int max_index =-1;//= get_max_element(next_state->occurrence);
        if(!unit_clauses.empty())
        {
       
#ifdef DEBUG
            CkPrintf("map size:%d", unit_clauses.size());
#endif
            int max_occur = -3;
            for(int _i=0; _i<next_state->occurrence.size(); _i++)
            {
                if(next_state->occurrence[_i]>max_occur&&(unit_clauses[_i+1]>0 || unit_clauses[-_i-1]>0))
                {
                    max_index = _i;
                    max_occur = next_state->occurrence[_i];
                }
            }
        }else 
            max_index = get_max_element(next_state->occurrence);
#ifdef DEBUG
        CkPrintf("max index = %d\n", max_index);
#endif
        if(next_state->occurrence[max_index] <=0)
        {
            CkPrintf("Unsatisfiable in sequential SAT\n");
            return false;
        }

        
        next_state->assigned_lit = Lit(max_index+1);
        next_state->assignclause(next_clauses);
        next_state->var_size = state_msg->var_size;
        next_state->level = state_msg->level+1;
        next_state->occurrence[max_index] = -2;

        bool satisfiable_1 = true;

        if(unit_clauses.empty()||unit_clauses[max_index+1]>0)        
        {
            satisfiable_1 = seq_solve(next_state);
            if(satisfiable_1)
            {
                return true;
            }
        }
        
        bool satisfiable_0 = true;
        next_state->assigned_lit = Lit(-max_index-1);
        next_state->occurrence[max_index] = -1;
        if(unit_clauses.empty()||unit_clauses[-max_index-1]>0)
        {
            satisfiable_0 = seq_solve(next_state);
            if(satisfiable_0)
            {
                return true;
            }
        }

        if(!satisfiable_1 && !satisfiable_0)
        {
            CkPrintf("Unsatisfiable through sequential\n");
            return false;
        }
    }

}
