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



static void parse_confFile(gzFile problem_stream, solution_stream) {                  
   
    vector<int> assignment;
    StreamBuffer in_solution(solution_stream);    
    
   
    for(;;)
    {
        if(*in == EOF)
            break;
        else
    }

    CkVec<Lit> lits;                                                 
    int i  = 0;
    
    for (;;)
    {                                                     
        skipWhitespace(in);                                        
        if (*in == EOF)                                            
            break;                                                 
        else if (*in == 'p'){                                      
            if (match(in, (char*)"pcnf")){                               
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

    if(msg->argc < 3)
    {
        error_exit((char*)"Usage: sat_verify problemfile solutionfile\n");
    }
    /* read file */

    /*read information from file */
    gzFile in_problem = gzopen(msg->argv[1], "rb");
    gzFile in_solution = gzopen(msg->argv[2], "rb");

    if(in == NULL)
    {
        error_exit((char*)"Invalid input filename\n");
    }

    parse_confFile(in, *solver_msg);

}

Main::Main(CkMigrateMessage* msg) {}

void Main::done()
{

    double endtimer = CmiWallTimer();

    CkPrintf("\nFile reading and processing time:         %f\n", readfiletimer-starttimer);
    CkPrintf("Solving time:                             %f\n", endtimer - readfiletimer);
    CkExit();
}
#include "main.def.h"

