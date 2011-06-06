#include "main.decl.h"

#include "searchEngine.h"
#include "exampleTsp.h"
#include <time.h>
#include <vector>
using namespace std;
int initial_grainsize;

CkVec< double> graph;

int  N;


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

void writeOutput()
{
    FILE *file;
    char filename[10];
    sprintf(filename, "%d.tsp", N);
    file = fopen(filename, "w");
    if(file == NULL)
    {
        printf("file write error %s\n", filename);
        CkExit();
    }
    fprintf(file, "%d", N);
    graph.resize(N*N);
    srand(time(0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            int edge = (int)( 1.1 * abs( (rand() % 701)) );
            graph[i*N+j] = edge;
            graph[j*N+i] = edge;
            //CkPrintf("%d\t", edge);
            fprintf(file, "%d ",edge); 
        }
        fprintf(file, "\n");
        //CkPrintf("\n");
    }
    fclose(file);

}
void readinput_2(char* filename)
{
    FILE *file;

    char line[128];
    char value[64];
    vector<double> xcoord;
    vector<double> ycoord;
    int start;
    int k, j;

    file = fopen(filename, "r");
    if(file == NULL)
    {
        printf("file read error %s\n", filename);
        CkExit();
    }
    /* parse the header lines to get the number of vertices*/
    fgets(line, 128, file);
    sscanf(line, "%d", &N);
    graph.resize(N*N);
    for(int i=1; i<N; i++)
    {
        fgets(line, 128, file);
        k = 0;
        start = 0;
        j=0;
        while(line[k]!='\0')
        {
            if(line[k]==' ')
            {
                memcpy(value, line+start, k-start); 
                value[k-start]='\0';
                //CkPrintf("value=%s, length=%d",  value, k-start);
                start = k+1;
                graph[i*N+j] = atof(value);
                graph[j*N+i] = atof(value);
                //CkPrintf("%f ", graph[i*N+j]);
                j++;
            }
            k++;
        }
        //CkPrintf("\n");
    }
    fclose(file);
}

void readinput(char* filename)
{
    FILE *file;

    char line[128];
    char variable[64];
    char value[64];
    vector<double> xcoord;
    vector<double> ycoord;

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
    int index;
    float x, y; 
    for(int i=0; i<N; i++)
    {
        fgets(line, sizeof line, file);

        sscanf(line, "%d %f %f", &index, &x, &y);
        xcoord[i] = x;
        ycoord[i] = y;
        CkPrintf("%d %f %f\n", index, x, y);
    }
    graph.resize(N*N);
    for(int i=0; i<N; i++)
        for(int j=0; j<i; j++)
        {
            double distance =  sqrt ((xcoord[i]-xcoord[j])*(xcoord[i]-xcoord[j]) + (ycoord[i]-ycoord[j])*(ycoord[i]-ycoord[j]));
            graph[i*N+j]= distance;
            graph[j*N+i]= distance;
        }
    CkPrintf("Done with reading\n");
    fclose(file);
}
 

extern void set_statesize(int);

class Main
{
public:
    Main(CkArgMsg* m )
    {

        int arg_index = 1;
        if(m->argc<2)
        {
            CkPrintf("Usage: tsp type(0-random graph, 1 inputfile, 2 inputfile) (Size of Problem) initialgrain\n");
            delete m;
            CkExit();
        }
        int type = atoi(m->argv[1]);
        if(type == 0)
        {
            N = atoi(m->argv[2]);
            writeOutput();
        }else if(type == 1) //read from file
        {
            /* input file */
            readinput(m->argv[2]);
        }else if(type == 2)
        {
            readinput_2(m->argv[2]);
        }
        initial_grainsize = atoi(m->argv[3]);

        set_statesize(N);

        CkPrintf("start\n");
        searchEngineProxy.start();
        delete m;
    }

};

#include "main.def.h"
