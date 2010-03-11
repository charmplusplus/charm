#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <errno.h>

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
    SolverState* solver_msg = new SolverState;
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
    /* simplify() */
    readfiletimer = CmiWallTimer();

    /*fire the first chare */
    /* 1)Which variable is assigned which value this time, (variable, 1), current clauses status vector(), literal array activities */

    mainProxy = thisProxy;
    int max_index = get_max_element(solver_msg->occurrence);
   
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
        CProxy_Solver::ckNew(solver_msg);
        CProxy_Solver::ckNew(not_msg);
    }else
    {
        CProxy_Solver::ckNew(not_msg);
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

