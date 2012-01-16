#include "main.decl.h"

#include "searchEngine.h"
#include <vector>

using namespace std;

int initial_grainsize;
char inputfile[50];

/* readonly */ int verticesNum;
/* readonly */ CkVec<int> inputGraph;

void readinput(char* filename)
{
    FILE *file;

    char line[128];
    char variable[64];
    char value[64];

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
            verticesNum = atoi(value);
        }else if(strncmp(line, "EDGE_DATA_SECTION", 17) == 0)
        {
            break;
        }
    }
   
    vector<int>  verticeNbs;
    verticeNbs.resize(verticesNum);

    /* get the edges, src dest */
    int src, dest;
    int previous=-1;
    int countptr = 0;
    int edgeNum = 0;
    while(fgets(line, sizeof line, file) != NULL && strncmp(line, "-1", 2) != 0)
    {
        edgeNum += 1;
        sscanf(line, "%d %d", &src, &dest);
        if(src != previous)
        {
        
            inputGraph.push_back(src-1);
            //CkPrintf("\nSource: %d %d:", src, inputGraph[countptr]);
            previous = src;
            countptr = inputGraph.size(); 
            inputGraph.push_back(1);
        }else
        {
            inputGraph[countptr]++;
        }
        //CkPrintf("  %d  ", dest);
        inputGraph.push_back(dest-1);
    }
#ifdef YHDEBUG
    CkPrintf("\n");
    for(int i=0; i<inputGraph.size(); i++)
    {
   
        CkPrintf(" %d ", inputGraph[i]);
    }

    CkPrintf("+++++++++++++===\n");
#endif
    fclose(file);

}

class Main
{
public:
    Main(CkArgMsg* msg )
    {
        if(msg->argc == 3)
        {
            strcpy(inputfile, msg->argv[1]);
            initial_grainsize = atoi(msg->argv[2]);
        }else
        {
            CkPrintf("exec inputfile grainsize\n");
            CkExit();
        }
        delete msg;
        
        readinput(inputfile);
        searchEngineProxy.start();
    }

};

#include "main.def.h"
