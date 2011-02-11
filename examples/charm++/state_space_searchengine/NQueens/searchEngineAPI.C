#ifndef __SEARCHENGINEAPI__
#define __SEARCHENGINEAPI__

#include "searchEngine.h"

/*   framework for search engine */


extern int initial_grainsize;
extern int numQueens;

class QueenStateBase : public StateBase
{
public:
    int aQueenBitRes;
    int aQueenBitCol;
    int aQueenBitPosDiag;
    int aQueenBitNegDiag;
    int numrows;
    
};

void createInitialChildren(Solver *solver)
{
    int odd = numQueens&1;
    int board_minus = numQueens - 1;
    int mask = (1<<numQueens) - 1;
    int half = numQueens >> 1;
    int numrows = 1;
    int col;
    int pos;
    int neg;
    int lsb;

    int bitfield  = (1 << half) -1;
    int childIndex = 0;
    for(;;)
    {
        if(bitfield == 0)
            break;
        lsb = -((signed)bitfield) & bitfield; 
        QueenStateBase *qmsg = (QueenStateBase*)solver->registerRootState(sizeof(QueenStateBase), childIndex, numQueens);
        qmsg->aQueenBitRes= lsb;
        qmsg->aQueenBitCol = 0|lsb;
        qmsg->aQueenBitNegDiag = (0|lsb) >> 1;
        qmsg->aQueenBitPosDiag = (0|lsb) << 1;
        qmsg->numrows = 1; 
        bitfield &= ~lsb;
        childIndex++;
        solver->process(qmsg);
    }

    if(odd == 1)
    {
        bitfield = 1 << (numQueens >> 1);
        numrows = 1; /* prob. already 0 */
        /* The first row just has one queen (in the middle column).*/
        col = bitfield;
        /* Now do the next row.  Only set bits in half of it, because we'll
         * * * * * * *                flip the results over the "Y-axis".  */
        neg = (bitfield >> 1);
        pos = (bitfield << 1);
        bitfield = (bitfield - 1) >> 1;
        for(;;)
        {
            if(bitfield == 0)
                break;
            lsb = -((signed)bitfield) & bitfield;

            QueenStateBase *qmsg = (QueenStateBase*)solver->registerRootState(sizeof(QueenStateBase), childIndex, numQueens);
            qmsg->aQueenBitRes = lsb;
            qmsg->aQueenBitCol = col|lsb;
            qmsg->aQueenBitNegDiag = (neg|lsb)>>1;
            qmsg->aQueenBitPosDiag = (pos|lsb)<<1;
            qmsg->numrows = 2; 
            bitfield &= ~lsb;
            childIndex++;
            solver->process(qmsg);
        }
    }
}


inline void createChildren( StateBase *_base , Solver* solver, bool parallel)
{
    register int lsb;
    QueenStateBase base = *((QueenStateBase*)_base);
    int n = base.numrows;
    register int bitfield;
    register int mask;
    int current_row = base.numrows+1;
    int childIndex = 0;

    mask = (1 << numQueens ) - 1;
    bitfield = mask & ~(base.aQueenBitCol | base.aQueenBitNegDiag | base.aQueenBitPosDiag);
    for(;;)
    {
        if(bitfield == 0)
        {
            break;
        }
        lsb = -((signed)bitfield)&bitfield;
        bitfield &= ~lsb;
        const int aQueenBitCol = base.aQueenBitCol | lsb;
        const int aQueenBitNegDiag = (base.aQueenBitNegDiag |lsb) >> 1;
        const int aQueenBitPosDiag = (base.aQueenBitPosDiag | lsb) << 1;
        const int new_bitfield = mask & ~(aQueenBitCol|aQueenBitPosDiag|aQueenBitNegDiag);
        
        if(current_row == numQueens-1 && new_bitfield != 0)
        {
            solver->reportSolution();
        }
        else if(new_bitfield == 0) {
        }
        else{
            QueenStateBase *new_branch  = (QueenStateBase*)solver->registerState(sizeof(QueenStateBase), childIndex, numQueens);
            new_branch->aQueenBitRes = lsb;
            new_branch->aQueenBitCol = aQueenBitCol;
            new_branch->aQueenBitNegDiag = aQueenBitNegDiag;
            new_branch->aQueenBitPosDiag = aQueenBitPosDiag;
            new_branch->numrows = current_row;
            if(parallel) {
                solver->process(new_branch);
            }
        }
    }

}

    inline double cost( )
    {
        return 0;
    }

    double heuristic( )
    {
        return 0;
    }

    double bound( int &l )
    {
        return 0;
    }

    //Search Engine Option
    int parallelLevel()
    {
        return initial_grainsize;
    }

    int searchDepthLimit()
    {
        return 1;
    }

    int minimumLevel()
    {
        return 1;
    }

    int maximumLevel()
    {
        return  numQueens;
    }
    inline void searchDepthChangeNotify( int ) {}


    SE_Register(QueenStateBase, createInitialChildren, createChildren, parallelLevel, searchDepthLimit);

#endif
