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

    /* timer */
    mainhandle=thishandle;
    
    
    int odd = numQueens&1;
    int board_minus = numQueens - 1;
    mask = (1<<numQueens) - 1;
    int bitfield = 0;
    int half = numQueens >> 1;
    int numrows = 1;
    int col;
    int pos;
    int neg;
    int lsb;

    bitfield  = (1 << half) -1;
    for(;;)
    {
        if(bitfield == 0)
            break;
        lsb = -((signed)bitfield) & bitfield; 
        QueenState *qmsg = new QueenState;
        qmsg->aQueenBitRes= lsb;
        qmsg->aQueenBitCol = 0|lsb;
        qmsg->aQueenBitNegDiag = (0|lsb) >> 1;
        qmsg->aQueenBitPosDiag = (0|lsb) << 1;
        qmsg->numrows = 1; 
        //CkSetQueueing(qmsg, CK_QUEUEING_ILIFO);
        CProxy_NQueen::ckNew(qmsg);
        bitfield &= ~lsb;
    }

    if(odd == 1)
    {
        bitfield = 1 << (numQueens >> 1);
        numrows = 1; /* prob. already 0 */
        /* The first row just has one queen (in the middle column).*/
        //aQueenBitRes[0] = bitfield;
        //aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0;
        col = bitfield;
        /* Now do the next row.  Only set bits in half of it, because we'll
         * * * *                flip the results over the "Y-axis".  */
        neg = (bitfield >> 1);
        pos = (bitfield << 1);
        bitfield = (bitfield - 1) >> 1;
        for(;;)
        {
            if(bitfield == 0)
                break;
            lsb = -((signed)bitfield) & bitfield; 

            QueenState *qmsg = new QueenState;
            qmsg->aQueenBitRes = lsb;
            qmsg->aQueenBitCol = col|lsb;
            qmsg->aQueenBitNegDiag = (neg|lsb)>>1;
            qmsg->aQueenBitPosDiag = (pos|lsb)<<1;
            qmsg->numrows = 2; 
            //CkSetQueueing(qmsg, CK_QUEUEING_ILIFO);
            CProxy_NQueen::ckNew(qmsg);
            bitfield &= ~lsb;
        }
    }
    starttimer = CkWallTimer();//CkTimer();
    counterGroup = counterInit();
    CkStartQD(CkIndex_Main::Quiescence1((DUMMYMSG *)0), &mainhandle);
}

Main::Main(CkMigrateMessage* msg) {}

void Main::Quiescence1(DUMMYMSG *msg)
{
    int numSolutions = CProxy_counter(counterGroup).ckLocalBranch()->getTotalCount();
    double endtimer = CkWallTimer();
    CkPrintf("There are %d Solutions to %d queens. Time=%f End time=%f\n",
        2*numSolutions, numQueens, endtimer-starttimer, CkWallTimer());
    CkExit();
}

#include "main.def.h"
