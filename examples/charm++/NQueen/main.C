/*
 * N Queen solver 
 * Feb 1st
 *
 * Copyright by PPL
 * problem report sun51@uiuc.edu
*/

#include "main.decl.h"
#include "main.h"
#include "nqueen.h"
#include "counter.h"

Main::Main(CkArgMsg* msg)
{

    //CkPrintf(" Usage: nqueen Num\n");

    numQueens = 5;
    grainsize = 3;
    if(msg->argc > 2)
    {
        numQueens = atoi(msg->argv[1]);
        grainsize = atoi(msg->argv[2]);   
    }
    delete msg;

    CkPrintf(" Usage: nqueen numQueen grainsize (%d, g=%d, np=%d)\n", numQueens, grainsize, CkNumPes());

    CkVec<int> initStatus(numQueens);
    for(int i=0; i<numQueens; i++)
        initStatus[i] = -1;

    /* timer */
    mainhandle=thishandle;
   
    QueenState *qmsg = new QueenState;
    qmsg->row = -1; 
    CProxy_NQueen::ckNew(qmsg);
    
    starttimer = CkTimer();
    counterGroup = counterInit();
    CkStartQD(CkIndex_Main::Quiescence1((DUMMYMSG *)0), &mainhandle);
}

Main::Main(CkMigrateMessage* msg) {}

void Main::Quiescence1(DUMMYMSG *msg)
{
    int numSolutions = CProxy_counter(counterGroup).ckLocalBranch()->getTotalCount();
    double endtimer = CkTimer();
    CkPrintf("There are %d Solutions to %d queens. Time=%f\n",
        numSolutions, numQueens, endtimer-starttimer);
    CkExit();
}

#include "main.def.h"
