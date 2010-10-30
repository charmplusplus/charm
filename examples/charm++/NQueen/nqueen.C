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
    current_row = m->row;
    QueenState *new_msg;
    /*for all possible places in next row */
    if(current_row == numQueens-1)
    {
        CProxy_counter(counterGroup).ckLocalBranch()->increment();
        //printSolution(m->queenInWhichColum);
        return;
    }

    bool conflict = false;
    for(int i_colm=0; i_colm< numQueens; i_colm++)
    {
       /* conflict check (row+1, i_colm)*/
        conflict = false;
        for(int j_row=0; j_row <= current_row; j_row++)
        {
            //conflict for the same colum
            if(m->queenInWhichColum[j_row] == i_colm || i_colm - m->queenInWhichColum[j_row] == (current_row+1 - j_row) || i_colm - m->queenInWhichColum[j_row] == -(current_row+1 - j_row) )
            {
                conflict = true;
                break;
            }
        }
        if(conflict){
            continue;
        }else
        {
             // the next step is reasonable. fire a new child
            if(numQueens - current_row > grainsize)
            {
                new_msg = new QueenState;
                new_msg->row = current_row+1;
                for(int __i=0; __i<current_row+1;__i++)
                    new_msg->queenInWhichColum[__i] = m->queenInWhichColum[__i];
                new_msg->queenInWhichColum[current_row+1] = i_colm; 
                CProxy_NQueen::ckNew(new_msg);
            }else
            {
                //sequential solve
                sequentialSolve(m->queenInWhichColum, current_row+1, i_colm);
            }
        }
    }
}

void NQueen::sequentialSolve(int queenInWhichColum[], int row, int colum)
{

    queenInWhichColum[row] = colum;
    if(row == numQueens -1){
        CProxy_counter(counterGroup).ckLocalBranch()->increment();
        //CkPrintf("[%d, %d]\n", row, colum);
        //printSolution(queenInWhichColum);
        return;
    }   
    bool conflict = false;
    for(int i_colm=0; i_colm< numQueens; i_colm++)
    {
        /* conflict check (row+1, i_colm)*/        /*all existing queens */
        conflict = false;
        for(int j_row=0; j_row <= row; j_row++)
        {
            //conflict for the same colum
            if(queenInWhichColum[j_row] == i_colm || i_colm-queenInWhichColum[j_row] == (row+1 - j_row) || i_colm - queenInWhichColum[j_row] == -(row+1 - j_row) )
            {
                conflict = true;
                break;
            }
        }
        // all colums are checked
        if(conflict){
            continue;
        }else
        {
            /* the last row , get  a solution. stop */
             /* the next step is reasonable. fire a new child */
                sequentialSolve(queenInWhichColum, row+1, i_colm);
        }

    }
}
void NQueen::printSolution(int queens[])
{
    for(int i=0; i<numQueens; i++)
    {
        CkPrintf("[%d, %d] ", i, queens[i]); 
    }
    CkPrintf("\n");
}

/*
void NQueen::pup(PUP::er &p)
{
    p | current_row;
    p | queenInWhichColum;
}
*/
