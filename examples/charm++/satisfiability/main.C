#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <errno.h>
#include <limits.h>

#include <signal.h>
#include <zlib.h>

#include "main.decl.h"
#include "main.h"

#include "SolverTypes.h"
#include "Solver.h"


//#include "util.h"

CProxy_Main mainProxy;

#define CHUNK_LIMIT 1048576


class StreamBuffer {
    gzFile  in;
    char    buf[CHUNK_LIMIT];
    int     pos;
    int     size;

    void assureLookahead() {
        if (pos >= size) {
            pos  = 0;
            size = gzread(in, buf, sizeof(buf)); } }

public:
            StreamBuffer(gzFile i) : in(i), pos(0), size(0) {
                assureLookahead(); }

            int  operator *  () { return (pos >= size) ? EOF : buf[pos]; }
            void operator ++ () { pos++; assureLookahead(); }
};

static bool match(StreamBuffer& in, char* str) {
    for (; *str != 0; ++str, ++in)
        if (*str != *in)
            return false;
    return true;
}


void error_exit(char *error)
{
    printf("%s\n", error);
    CkExit();
}

void skipWhitespace(StreamBuffer& in) 
{
    while ((*in >= 9 && *in <= 13) || *in == 32)
        ++in;
}

void skipLine(StreamBuffer& in) {
    for (;;){
        if (*in == EOF || *in == '\0') return;
        if (*in == '\n')
        { ++in; return; }
        ++in;
    } 
}


int parseInt(StreamBuffer& in) {
    int     val = 0;
    bool    neg = false;
    skipWhitespace(in);
    if      (*in == '-') neg = true, ++in;
    else if (*in == '+') ++in;
    if (*in < '0' || *in > '9')
        error_exit((char*)"ParseInt error\n");

    while (*in >= '0' && *in <= '9')
    {
        val = val*10 + (*in - '0');
        ++in;
    }
    return neg ? -val : val; 
}

static void readClause(StreamBuffer& in, SolverState& S, CkVec<Lit>& lits) {
    int     parsed_lit, var;
    lits.removeAll();
    for (;;){
        parsed_lit = parseInt(in);
        if (parsed_lit == 0) break;
        var = abs(parsed_lit)-1;
        
        S.occurrence[var]++;
        if(parsed_lit>0)
            S.positive_occurrence[var]++;
        //while (var >= S.nVars()) S.newVar();
        lits.push_back( Lit(parsed_lit));
    }
}

/* unit propagation before real computing */

static void simplify(SolverState& S)
{
    for(int i=0; i< S.unit_clause_index.size(); i++)
    {
#ifdef DEBUG
        CkPrintf("Inside simplify before processing, unit clause number:%d, i=%d\n", S.unit_clause_index.size(), i);
#endif
       
        Clause cl = S.clauses[S.unit_clause_index[i]];
        //only one element in unit clause
        Lit lit = cl[0];
        S.clauses[S.unit_clause_index[i]].resize(0);
        S.unsolved_clauses--;

        int pp_ = 1;
        int pp_i_ = 2;
        int pp_j_ = 1;

       if(toInt(lit) < 0)
       {
           pp_ = -1;
           pp_i_ = 1;
           pp_j_ = 2;
       }
       S.occurrence[pp_*toInt(lit)-1] = -pp_i_;
       CkVec<int> &inClauses = S.whichClauses[pp_*2*toInt(lit)-pp_i_];
       CkVec<int> &inClauses_opposite = S.whichClauses[pp_*2*toInt(lit)-pp_j_];
       
#ifdef DEBUG
        CkPrintf("****\nUnit clause is %d, index=%d, occurrence=%d\n", toInt(lit), S.unit_clause_index[i], -pp_i_);
           CkPrintf("\t same occur");
#endif
       // literal with same sign
       for(int j=0; j<inClauses.size(); j++)
       {
           int cl_index = inClauses[j];
#ifdef DEBUG
           CkPrintf(" %d \n \t \t literals in this clauses: ", cl_index);
#endif
           Clause& cl_ = S.clauses[cl_index];
           //for all the literals in this clauses, the occurrence decreases by 1
           for(int k=0; k< cl_.size(); k++)
           {
               Lit lit_ = cl_[k];
               if(toInt(lit_) == toInt(lit))
                   continue;
#ifdef DEBUG
               CkPrintf(" %d  ", toInt(lit_));
#endif
               S.occurrence[abs(toInt(lit_)) - 1]--;
               if(toInt(lit_) > 0)
               {
                   S.positive_occurrence[toInt(lit_)-1]--;
                   //S.whichClauses[2*toInt(lit_)-2].remove(cl_index);
                //remove the clause index for the literal
               for(int _i = 0; _i<S.whichClauses[2*toInt(lit_)-2].size(); _i++)
                   {
                       if(S.whichClauses[2*toInt(lit_)-2][_i] == cl_index)
                       {
                           S.whichClauses[2*toInt(lit_)-2].remove(_i);
                            break;
                       }
                   }

               }else
               {
                   for(int _i = 0; _i<S.whichClauses[-2*toInt(lit_)-1].size(); _i++)
                   {
                       if(S.whichClauses[-2*toInt(lit_)-1][_i] == cl_index)
                       {
                           S.whichClauses[-2*toInt(lit_)-1].remove(_i);
                           break;
                       }
                   }
               }
           }
           
           S.unsolved_clauses--;
           S.clauses[cl_index].resize(0); //this clause goes away. In order to keep index unchanged, resize it as 0
       }
       
       for(int j=0; j<inClauses_opposite.size(); j++)
       {
           int cl_index_ = inClauses_opposite[j];
           Clause& cl_neg = S.clauses[cl_index_];
           cl_neg.remove(-toInt(lit));
           //becomes a unit clause
           if(cl_neg.size() == 1)
           {
               S.unit_clause_index.push_back(cl_index_);
           }
       }

    }

    S.unit_clause_index.removeAll();
}



static void parse_confFile(gzFile input_stream, SolverState& S) {                  
    StreamBuffer in(input_stream);    
      CkVec<Lit> lits;                                                 
    int i  = 0;
    
    for (;;){                                                      
        //printf(" + on %d\n", i++);
        skipWhitespace(in);                                        
        if (*in == EOF)                                            
            break;                                                 
        else if (*in == 'p'){                                      
            if (match(in, (char*)"p cnf")){                               
                int vars    = parseInt(in);                        
                int clauses = parseInt(in);                        
                printf("|  Number of variables:  %-12d                                         |\n", vars);
                printf("|  Number of clauses:    %-12d                                         |\n", clauses);
      
                S.var_size = vars;
                S.occurrence.resize(vars);
                S.positive_occurrence.resize(vars);
                S.whichClauses.resize(2*vars);
                for(int __i=0; __i<vars; __i++)
                {
                    S.occurrence[__i] = 0;
                    S.positive_occurrence[__i] = 0;
                }
            }else{
                printf("PARSE ERROR! Unexpected char: %c\n", *in);
                error_exit((char*)"Parse Error\n");
            }
        } else if (*in == 'c' || *in == 'p')
            skipLine(in);
        else{
            readClause(in, S, lits);
            if( !S.addClause(lits))
            {
                CkPrintf("conflict detected by addclauses\n");
                CkExit();
            }
            
        }
    
    }
}


Main::Main(CkArgMsg* msg)
{

    grainsize = 1;
    SolverState* solver_msg = new (8 * sizeof(int))SolverState;
    if(msg->argc < 2)
    {
        error_exit((char*)"Usage: sat filename grainsize\n");
    }else
        grainsize = atoi(msg->argv[2]);

    char filename[50];

    /* read file */

    starttimer = CmiWallTimer();

    /*read information from file */
    gzFile in = gzopen(msg->argv[1], "rb");

    if(in == NULL)
    {
        error_exit((char*)"Invalid input filename\n");
    }

    parse_confFile(in, *solver_msg);


    /*  unit propagation */ 
    simplify(*solver_msg);

#ifdef DEBUG
    for(int __i = 0; __i<solver_msg->occurrence.size(); __i++)
    {

            if(solver_msg->occurrence[__i] == -2)
                CkPrintf(" TRUE ");
            else if(solver_msg->occurrence[__i] == -1)
                CkPrintf(" FALSE ");
            else
                CkPrintf(" UNDECIDED ");
    }


    CkPrintf(" unsolved clauses %d\n", solver_msg->unsolved_clauses);
#endif

    solver_msg->unsolved_clauses = 0;
    for(int __i=0; __i<solver_msg->clauses.size(); __i++)
    {
        if(solver_msg->clauses[__i].size() > 0)
            solver_msg->unsolved_clauses++;
    }
    
    readfiletimer = CmiWallTimer();
    /*fire the first chare */
    /* 1)Which variable is assigned which value this time, (variable, 1), current clauses status vector(), literal array activities */

    mainProxy = thisProxy;
    int max_index = get_max_element(solver_msg->occurrence);
  
    if(max_index < 0)
    {
        CkPrintf(" This problem is solved by pre-processing\n");
        CkExit();
    }
    solver_msg->assigned_lit = Lit(max_index+1);
    solver_msg->level = 0;
    SolverState *not_msg = copy_solverstate(solver_msg);
    
    //CkPrintf(" main chare max index=%d, %d, assigned=%d\n", max_index+1, solver_msg->occurrence[max_index], toInt(solver_msg->assigned_lit));
    solver_msg->occurrence[max_index] = -2;
    not_msg->assigned_lit = Lit(-max_index-1);
    not_msg->occurrence[max_index] = -1;
    
    int positive_max = solver_msg->positive_occurrence[max_index];
    if(positive_max >= solver_msg->occurrence[max_index] - positive_max)
    {

        // assign true first and then false
        *((int *)CkPriorityPtr(solver_msg)) = INT_MIN;
        CkSetQueueing(solver_msg, CK_QUEUEING_IFIFO);
        solver_msg->lower = INT_MIN;
        solver_msg->higher = 0;
        CProxy_Solver::ckNew(solver_msg);
        
        *((int *)CkPriorityPtr(not_msg)) = 0;
        CkSetQueueing(not_msg, CK_QUEUEING_IFIFO);
        not_msg->lower = 0;
        not_msg->higher = INT_MAX;
        CProxy_Solver::ckNew(not_msg);
    }else
    {
        *((int *)CkPriorityPtr(not_msg)) = INT_MIN;
        CkSetQueueing(not_msg, CK_QUEUEING_IFIFO);
        not_msg->lower = INT_MIN;
        not_msg->higher = 0;
        CProxy_Solver::ckNew(not_msg);
        
        *((int *)CkPriorityPtr(solver_msg)) = 0;
        CkSetQueueing(solver_msg, CK_QUEUEING_IFIFO);
        solver_msg->lower = 0;
        solver_msg->higher = INT_MAX;
        CProxy_Solver::ckNew(solver_msg);

    }
}

Main::Main(CkMigrateMessage* msg) {}

void Main::done()
{

    double endtimer = CmiWallTimer();

    CkPrintf("File reading and processing time:         %f\n", readfiletimer-starttimer);
    CkPrintf("Solving time:                             %f\n", endtimer - readfiletimer);
    CkExit();
}
#include "main.def.h"

