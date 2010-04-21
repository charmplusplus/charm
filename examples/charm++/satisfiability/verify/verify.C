#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <errno.h>
#include <limits.h>

#include <signal.h>
#include <zlib.h>

#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <vector>
#include <signal.h>
#include <zlib.h>
#include <fstream>
#include <iostream>


#define CHUNK_LIMIT 1048576

using namespace std;

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
    exit(1);
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


static void parse_confFile( char* solution_file, gzFile problem_stream) {                 

    int var_size;
    StreamBuffer in(problem_stream);
    vector<int> assignment;

    ifstream fin(solution_file);
    char assign[5];

    while(!fin.eof())
    {
        fin.getline(assign, 5);
        assignment.push_back(atoi(assign));    
    }
 
    assignment.pop_back();
    fin.close();

    int i  = 0;
    for (;;)
    {                                                     
        skipWhitespace(in);                        
        i++;
        if (*in == EOF)                                            
            break;                                                 
        else if (*in == 'p'){                                      
            if (match(in, (char*)"p cnf")){                               
                int vars    = parseInt(in);                        
                int clauses = parseInt(in);                        
                printf("|  Number of variables:  %-12d                                         |\n", vars);
                printf("|  Number of clauses:    %-12d                                         |\n", clauses);

                var_size = vars;
            }else{
                printf("PARSE ERROR! Unexpected char: %c\n", *in);
                error_exit((char*)"Parse Error\n");
            }
        } else if (*in == 'c' || *in == 'p')
            skipLine(in);
        else{
            
            int     parsed_lit;
            bool satisfied = false;
            for (;;){
                parsed_lit = parseInt(in);

                if(parsed_lit>0 && (assignment[parsed_lit-1]==-2 || assignment[parsed_lit-1]==0))
                {
                    satisfied = true;
                }
                if(parsed_lit<0 && (assignment[-parsed_lit-1]==-1 || assignment[-parsed_lit-1]==0))
                {
                    satisfied = true;
                }

                if (parsed_lit == 0)
                {
                    if(satisfied)
                        break;
                    else
                    {
                        printf("Result unreasonable, check line %d\n", i);
                        exit(0);
                    };
                }
            }
        }
    }
}


int main(int argc, char* argv[])
{

    if(argc < 3)
    {
        error_exit((char*)"Usage: sat_verify problemfile solutionfile\n");
    }
    /* read file */

    /*read information from file */
    gzFile in_problem = gzopen(argv[1], "rb");
    if(in_problem == NULL)
    {
        error_exit((char*)"Invalid input filename\n");
    }

    parse_confFile(argv[2], in_problem);

    printf("This result is correct\n");

    return 0;
}
