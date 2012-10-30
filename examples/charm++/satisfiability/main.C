#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <errno.h>
#include <limits.h>

#include <signal.h>
#include <zlib.h>

#include <vector>

#include "main.decl.h"
#include "main.h"

#include "par_SolverTypes.h"
#include "par_Solver.h"

#ifdef MINISAT
#include "Solver.h"
#endif

#ifdef TNM
#include "TNM.h"
#endif

using namespace std;

//#include "util.h"

typedef map<int, int> map_int_int;

CProxy_Main mainProxy;

#define CHUNK_LIMIT 1048576

char inputfile[50];

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

static void readClause(StreamBuffer& in, par_SolverState& S, CkVec<par_Lit>& lits) {
    int     parsed_lit, var;
    lits.removeAll();
    for (;;){
        parsed_lit = parseInt(in);
        if (parsed_lit == 0) break;
        var = abs(parsed_lit)-1;
        
        S.occurrence[var]++;
        if(parsed_lit>0)
            S.positive_occurrence[var]++;
        lits.push_back( par_Lit(parsed_lit));
    }
}

/* unit propagation before real computing */

static void simplify(par_SolverState& S)
{
    for(int i=0; i< S.unit_clause_index.size(); i++)
    {
#ifdef DEBUG
        CkPrintf("Inside simplify before processing, unit clause number:%d, i=%d\n", S.unit_clause_index.size(), i);
#endif
       
        par_Clause cl = S.clauses[S.unit_clause_index[i]];
        //only one element in unit clause
        par_Lit lit = cl[0];
        S.clauses[S.unit_clause_index[i]].resize(0);

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
       map_int_int &inClauses = S.whichClauses[pp_*2*toInt(lit)-pp_i_];
       map_int_int &inClauses_opposite = S.whichClauses[pp_*2*toInt(lit)-pp_j_];
       
       // literal with same sign
       for( map_int_int::iterator iter = inClauses.begin(); iter!=inClauses.end(); iter++)
       {
           int cl_index = (*iter).first;
#ifdef DEBUG
           CkPrintf(" %d \n \t \t literals in this clauses: ", cl_index);
#endif
           par_Clause& cl_ = S.clauses[cl_index];
           //for all the literals in this clauses, the occurrence decreases by 1
           for(int k=0; k< cl_.size(); k++)
           {
               par_Lit lit_ = cl_[k];
               if(toInt(lit_) == toInt(lit))
                   continue;
#ifdef DEBUG
               CkPrintf(" %d  ", toInt(lit_));
#endif
               S.occurrence[abs(toInt(lit_)) - 1]--;
               if(toInt(lit_) > 0)
               {
                   S.positive_occurrence[toInt(lit_)-1]--;
                   map_int_int::iterator one_it = S.whichClauses[2*toInt(lit_)-2].find(cl_index);
                   S.whichClauses[2*toInt(lit_)-2].erase(one_it);
               }else
               {
                   map_int_int::iterator one_it = S.whichClauses[-2*toInt(lit_)-1].find(cl_index);
                   S.whichClauses[-2*toInt(lit_)-1].erase(one_it);
               }

           }
           
           S.clauses[cl_index].resize(0); //this clause goes away. In order to keep index unchanged, resize it as 0
       }
       
       for(map_int_int::iterator iter= inClauses_opposite.begin(); iter!=inClauses_opposite.end(); iter++)
       {
           int cl_index_ = (*iter).first;
           par_Clause& cl_neg = S.clauses[cl_index_];
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



static void parse_confFile(gzFile input_stream, par_SolverState& S) {                  
    StreamBuffer in(input_stream);    
      CkVec<par_Lit> lits;                                                 
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
    par_SolverState* solver_msg = new (8 * sizeof(int))par_SolverState;
    if(msg->argc < 2)
    {
        error_exit((char*)"Usage: sat filename grainsize\n");
    }else
        grainsize = atoi(msg->argv[2]);


    CkPrintf("problem file:\t\t%s\ngrainsize:\t\t%d\nprocessor number:\t\t%d\n", msg->argv[1], grainsize, CkNumPes()); 

    /* read file */

    starttimer = CkWallTimer();

    /*read information from file */
    gzFile in = gzopen(msg->argv[1], "rb");

    strcpy(inputfile, msg->argv[1]);

    if(in == NULL)
    {
        error_exit((char*)"Invalid input filename\n");
    }

    parse_confFile(in, *solver_msg);

    solver_msg->printSolution();
    /*  unit propagation */ 
    simplify(*solver_msg);
#ifdef DEBUG
    for(int __i = 0; __i<solver_msg->occurrence.size(); __i++)
    {
        FILE *file;
        char outputfile[50];
        sprintf(outputfile, "%s.sat", inputfile);
        file = fopen(outputfile, "w");
        for(int i=0; i<assignment.size(); i++)
        {
            fprintf(file, "%d\n", assignment[i]);
        }
    }

#endif

    int unsolved = solver_msg->unsolvedClauses();

    if(unsolved == 0)
    {
        CkPrintf(" This problem is solved by pre-processing\n");
        CkExit();
    }
    readfiletimer = CkWallTimer();
    /*fire the first chare */
    /* 1)Which variable is assigned which value this time, (variable, 1), current clauses status vector(), literal array activities */


    /***  If grain size is larger than the clauses size, that means 'sequential' */
    if(grainsize > solver_msg->clauses.size())
    {
        vector< vector<int> > seq_clauses;
        for(int _i_=0; _i_<solver_msg->clauses.size(); _i_++)
        {
            if(solver_msg->clauses[_i_].size() > 0)
            {
                vector<int> unsolvedclaus;
                par_Clause& cl = solver_msg->clauses[_i_];
                unsolvedclaus.resize(cl.size());
                for(int _j_=0; _j_<cl.size(); _j_++)
                {
                    unsolvedclaus[_j_] = toInt(cl[_j_]);
                }
                seq_clauses.push_back(unsolvedclaus);
            }
        }
        bool satisfiable_1 = seq_processing(solver_msg->var_size, seq_clauses);//seq_solve(next_state);

        if(satisfiable_1)
        {
            CkPrintf("One solution found without using any parallel\n");
        }else
        {
       
            CkPrintf(" Unsatisfiable\n");
        }
        done(solver_msg->occurrence);
        return;
    }
    mainProxy = thisProxy;
    int max_index = get_max_element(solver_msg->occurrence);
    
    solver_msg->assigned_lit = par_Lit(max_index+1);
    solver_msg->level = 0;
    par_SolverState *not_msg = copy_solverstate(solver_msg);
    
    solver_msg->occurrence[max_index] = -2;
    not_msg->assigned_lit = par_Lit(-max_index-1);
    not_msg->occurrence[max_index] = -1;
    
    int positive_max = solver_msg->positive_occurrence[max_index];
    if(positive_max >= solver_msg->occurrence[max_index] - positive_max)
    {

        // assign true first and then false
        *((int *)CkPriorityPtr(solver_msg)) = INT_MIN;
        CkSetQueueing(solver_msg, CK_QUEUEING_IFIFO);
        solver_msg->lower = INT_MIN;
        solver_msg->higher = 0;
        CProxy_mySolver::ckNew(solver_msg);
        
        *((int *)CkPriorityPtr(not_msg)) = 0;
        CkSetQueueing(not_msg, CK_QUEUEING_IFIFO);
        not_msg->lower = 0;
        not_msg->higher = INT_MAX;
        CProxy_mySolver::ckNew(not_msg);
    }else
    {
        *((int *)CkPriorityPtr(not_msg)) = INT_MIN;
        CkSetQueueing(not_msg, CK_QUEUEING_IFIFO);
        not_msg->lower = INT_MIN;
        not_msg->higher = 0;
        CProxy_mySolver::ckNew(not_msg);
        
        *((int *)CkPriorityPtr(solver_msg)) = 0;
        CkSetQueueing(solver_msg, CK_QUEUEING_IFIFO);
        solver_msg->lower = 0;
        solver_msg->higher = INT_MAX;
        CProxy_mySolver::ckNew(solver_msg);

    }
}

Main::Main(CkMigrateMessage* msg) {}

void Main::done(CkVec<int> assignment)
{

    double endtimer = CkWallTimer();

    CkPrintf("\nFile reading and processing time:         %f\n", readfiletimer-starttimer);
    CkPrintf("Solving time:                             %f\n", endtimer - readfiletimer);
 
    FILE *file;
    char outputfile[50];
    sprintf(outputfile, "%s.sat", inputfile);
    file = fopen(outputfile, "w");

    for(int i=0; i<assignment.size(); i++)
    {
        fprintf(file, "%d\n", assignment[i]);
    }
    CkExit();
}
#include "main.def.h"

