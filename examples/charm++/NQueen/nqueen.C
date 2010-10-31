#include "main.decl.h"
#include "nqueen.h"
#include "counter.h"

extern int numQueens;
extern CProxy_Main mainProxy;
extern int grainsize;
extern CkGroupID counterGroup;

NQueen::NQueen(CkMigrateMessage *msg) {}

NQueen::NQueen(QueenState *m)
{
    int lsb;
    int n = m->numrows;
    int bitfield;// = m->bitfield;
    int mask; 
    int current_row = m->numrows+1;
    QueenState *new_msg;
    
    mask = (1 << numQueens ) - 1;
    bitfield = mask & ~(m->aQueenBitCol | m->aQueenBitNegDiag | m->aQueenBitPosDiag);

    /*for all possible places in next row */
    if(n == numQueens-1)
    {
        if(bitfield!=0){
            CProxy_counter(counterGroup).ckLocalBranch()->increment();
        }
            return;
    }
    // the next step is reasonable. fire a new child
    for(;;)
    {
    
        if(bitfield == 0)
        {
            break;
        }
        lsb = -((signed)bitfield)&bitfield;
        new_msg = new QueenState;
        new_msg->aQueenBitRes = lsb;
        new_msg->aQueenBitCol = m->aQueenBitCol | lsb;
        new_msg->aQueenBitNegDiag = (m->aQueenBitNegDiag |lsb) >> 1;
        new_msg->aQueenBitPosDiag = (m->aQueenBitPosDiag | lsb) << 1;
        new_msg->numrows = current_row;
        bitfield &= ~lsb;
        if(current_row < grainsize)
        {
            //CkSetQueueing(new_msg, CK_QUEUEING_ILIFO);
            CProxy_NQueen::ckNew(new_msg);
        }else
        {
            //sequential solve
            //thisProxy.sequentialSolve(new_msg);
            sequentialSolve(new_msg);
        }
    }
    //CkPrintf(" level = %d\n", current_row+1);
}

void NQueen::sequentialSolve( QueenState *m)
{

    int aQueenBitRes[MAX_BOARDSIZE];
    int aQueenBitCol[MAX_BOARDSIZE];
    int aQueenBitPosDiag[MAX_BOARDSIZE];
    int aQueenBitNegDiag[MAX_BOARDSIZE];
    int aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    int n = m->numrows;
    register int* pnStack;
    register int numrows = m->numrows; /* numrows redundant - could use stack */
    register unsigned int lsb; /* least significant bit */
    register unsigned int bitfield; /* bits which are set mark possible positions for a queen */
    int board_minus = numQueens - 1; /* board size - 1 */
    int mask = (1 << numQueens) - 1; /* if board size is N, mask consists of N 1's */
    aStack[0] = -1;
    
    pnStack = aStack + 1;

    aQueenBitCol[n] = m->aQueenBitCol;
    aQueenBitNegDiag[n] = m->aQueenBitNegDiag;
    aQueenBitPosDiag[n] = m->aQueenBitPosDiag;
    bitfield = mask & ~(m->aQueenBitCol | m->aQueenBitNegDiag | m->aQueenBitPosDiag);
    /* this is the critical loop */
    for (;;)
    {
        /* could use 
         * lsb = bitfield ^ (bitfield & (bitfield -1)); 
         * to get first (least sig) "1" bit, but that's slower. */
        lsb = -((signed)bitfield) & bitfield; /* this assumes a 2's complement architecture */
        if (0 == bitfield)
        {
            bitfield = *--pnStack; /* get prev. bitfield from stack */
            if (pnStack == aStack) { /* if sentinel hit.... */
                break ;
            }
            --numrows;
            continue;
        }
        bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

        aQueenBitRes[numrows] = lsb; /* save the result */
        if (numrows < board_minus) /* we still have more rows to process? */
        {
            n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
            *pnStack++ = bitfield;
            /* We can't consider positions for the queen which are in the same
             * * column, same positive diagonal, or same negative diagonal as another
             * queen already on the board. */
            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            continue;
        }
        else{
            /* printtable(board_size, aQueenBitRes, g_numsolutions + 1);  */
            CProxy_counter(counterGroup).ckLocalBranch()->increment();
            bitfield = *--pnStack;
            --numrows;
            continue;
        }
    }

}
void NQueen::printSolution(int queens[])
{
}

/*
void NQueen::pup(PUP::er &p)
{
    p | current_row;
    p | queenInWhichColum;
}
*/
