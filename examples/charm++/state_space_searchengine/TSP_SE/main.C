#include "main.decl.h"

#include "searchEngine.h"
#include "exampleTsp.h"

#include <vector>
using namespace std;
int initial_grainsize;

CkVec< int > graph;

int  N, maxD;


#undef M_PI
#define M_PI 3.14159265358979323846264
/*
int geom_edgelen (int i, int j, CCdatagroup *dat)
{
     double lati, latj, longi, longj;
     double q1, q2, q3, q4, q5;

     lati = M_PI * dat->x[i] / 180.0;
     latj = M_PI * dat->x[j] / 180.0;

     longi = M_PI * dat->y[i] / 180.0;
     longj = M_PI * dat->y[j] / 180.0;

     q1 = cos (latj) * sin(longi - longj);
     q3 = sin((longi - longj)/2.0);
     q4 = cos((longi - longj)/2.0);
     q2 = sin(lati + latj) * q3 * q3 - sin(lati - latj) * q4 * q4;
     q5 = cos(lati - latj) * q4 * q4 - cos(lati + latj) * q3 * q3;
     return (int) (6378388.0 * atan2(sqrt(q1*q1 + q2*q2), q5) + 1.0);
}
*/
void readinput(char* filename)
{
    FILE *file;

    char line[128];
    char variable[64];
    char value[64];
    vector<int> xcoord;
    vector<int> ycoord;

    file = fopen(filename, "r");
    if(file == NULL)
    {
        printf("file read error %s\n", filename);
        CkExit();
    }
    /* parse the header lines to get the number of vertices*/
    while(fgets(line, 128, file) != NULL)
    {
        if(strncmp(line, "DIMENSION", 9) == 0)
        {
            sscanf(line, "%s : %s", variable, value);
            N= atoi(value);
        }else if(strncmp(line, "NODE_COORD_SECTION", 18) == 0)
        {
            break;
        }
    }

    xcoord.resize(N);
    ycoord.resize(N);
    int index, x, y; 
    for(int i=0; i<N; i++)
    {
        fgets(line, sizeof line, file);

        sscanf(line, "%d %d %d", &index, &x, &y);
        xcoord[i] = x;
        ycoord[i] = y;
    }
    graph.resize(N*N);
    for(int i=0; i<N; i++)
        for(int j=0; j<i; j++)
        {
            int distance = (int) sqrt ((xcoord[i]-xcoord[j])*(xcoord[i]-xcoord[j]) + (ycoord[i]-ycoord[j])*(ycoord[i]-ycoord[j]));
            graph[i*N+j]= distance;
            graph[j*N+i]= distance;
        }
}
 

class Main
{
public:
    Main(CkArgMsg* m )
    {

        int arg_index = 1;
        if(m->argc<2)
        {
            CkPrintf("Usage: tsp type(0-random graph, 1 inputfile) (Size of Problem) (Maximum Distance) grain\n");
            delete m;
            CkExit();
        }
        int type = atoi(m->argv[1]);
        if(type == 0)
        {
            // Input Problem parameters
            N = atoi(m->argv[2]);
            maxD = atoi(m->argv[3]);
            initial_grainsize = atoi(m->argv[4]);

            graph.resize(N*N);
            // Allocate 2-D array for storing the distance
            // // Asymmetric Problem
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < i; j++) {
                    int edge = (int)( 1.1 * abs( (rand() % maxD)) );
                    if(edge >= maxD)
                    { // no edge
                        graph[i*N+j]= maxD;
                        graph[j*N+i]= maxD;
                    }else
                    {
                        graph[i*N+j] = edge;
                        graph[j*N+i] = edge;
                    }
                    CkPrintf("%d\t", graph[i*N+j]);
				}
                CkPrintf("\n");
			}
        }else if(type == 1) //read from file
        {
            /* input file */
            readinput(m->argv[2]);
            initial_grainsize = atoi(m->argv[3]);
        }

				CkPrintf("start\n");
        searchEngineProxy.start();
        delete m;
    }

};

#include "main.def.h"
